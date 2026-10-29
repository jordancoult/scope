import logging
import os
import time
from typing import TYPE_CHECKING

import torch
from diffusers.modular_pipelines import PipelineState

from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..defaults import (
    apply_mode_defaults_to_state,
    handle_mode_transition,
    prepare_for_mode,
    resolve_input_mode,
)
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..utils import Quantization, load_model_config, validate_resolution
from ..wan2_1.modules import attention_steer
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.vace import VACEEnabledPipeline
from ..wan2_1.vae import create_vae
from .components import StreamDiffusionV2WanVAEWrapper
from .modular_blocks import StreamDiffusionV2Blocks
from .schema import StreamDiffusionV2Config

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [750, 250]


class StreamDiffusionV2Pipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return StreamDiffusionV2Config

    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        stage_callback=None,
    ):
        from .modules.causal_model import CausalWanModel

        # Validate resolution requirements
        # VAE downsample (8) * patch embedding downsample (2) = 16
        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=16,
        )

        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        model_config = load_model_config(config, __file__)
        # Local patch (ai-character-scope): shorter KV window. On RTX 5080 tests 3 frames was
        # faster (fewer keys per attention call) and *less* smeary than the stock 6 with 1-step
        # denoising; the first chunk (where the VACE reference enters) stays as the sink.
        if os.environ.get("SCOPE_KV_FRAMES"):
            model_config.local_attn_size = int(os.environ["SCOPE_KV_FRAMES"])
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator"
        )

        # Load generator with VACE support via upfront loading
        if stage_callback:
            stage_callback("Loading diffusion model...")
        start = time.time()

        # Always create base CausalWanModel first
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )

        # Apply VACE wrapper if vace_path is configured (upfront loading)
        # This must happen before LoRA to get correct ordering: LoRA -> VACE -> Base
        generator.model = self._init_vace(
            config, generator.model, device=device, dtype=dtype
        )

        # Initialize optional LoRA adapters on the underlying model.
        generator.model = self._init_loras(config, generator.model)

        if quantization == Quantization.FP8_E4M3FN:
            # Cast before optional quantization
            generator = generator.to(dtype=dtype)

            start = time.time()

            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )

            # Move to target device during quantization
            # Defaults to using fp8_e4m3fn for both weights and activations
            quantize_(
                generator,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                device=device,
            )

            print(f"Quantized diffusion model to fp8 in {time.time() - start:.3f}s")
        else:
            generator = generator.to(device=device, dtype=dtype)

        # Local patch (ai-character-scope): Blackwell/Ada speedups, opt-in via env vars.
        if os.environ.get("SCOPE_FP4_FFN") and quantization is None:
            # Experimental: NVFP4 FFN (+~14% FPS on RTX 5080) but visible ghosting; "1" = up+down,
            # "2" = down-projection only. Runs before fp8 so FP4 is quantized from bf16 weights.
            from ..wan2_1.fp4_linear import convert_ffn_to_fp4

            layers = (2,) if os.environ["SCOPE_FP4_FFN"] == "2" else (0, 2)
            n = convert_ffn_to_fp4(generator.model, layers=layers)
            logger.info(f"SCOPE_FP4_FFN: {n} FFN linear layers -> NVFP4 (experimental)")
        if os.environ.get("SCOPE_FP8_ROWWISE") == "1" and quantization is None:
            from ..wan2_1.fp8_rowwise import convert_blocks_to_fp8_rowwise

            n = convert_blocks_to_fp8_rowwise(generator.model)
            torch.cuda.empty_cache()
            logger.info(f"SCOPE_FP8_ROWWISE: {n} linear layers -> fp8 row-wise _scaled_mm")
        if os.environ.get("SCOPE_COMPILE_BLOCKS") == "1":
            import torch._dynamo as dynamo

            from .modules import causal_model

            # Self-attention reads KV-cache positions as Python ints; keep it eager and
            # compile the rest of each block (norms, modulation, FFN, cross-attn).
            dynamo.config.recompile_limit = 1024
            dynamo.config.accumulated_recompile_limit = 4096
            dynamo.config.allow_unspec_int_on_nn_module = True
            attn_cls = causal_model.CausalWanSelfAttention
            if not getattr(attn_cls.forward, "_scope_eager", False):
                attn_cls.forward = torch.compiler.disable(attn_cls.forward)
                attn_cls.forward._scope_eager = True
            n = 0
            for m in generator.model.modules():
                if type(m).__name__.endswith("AttentionBlock"):
                    m.compile(fullgraph=False, dynamic=None)
                    n += 1
            logger.info(f"SCOPE_COMPILE_BLOCKS: compiling {n} transformer blocks (first chunks are slow)")

        if stage_callback:
            stage_callback("Loading text encoder...")
        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:.3f}s")
        # Move text encoder to target device but use dtype of weights
        text_encoder = text_encoder.to(device=device)

        # Load VAE using unified WanVAEWrapper
        if stage_callback:
            stage_callback("Loading VAE...")
        start = time.time()
        # Local patch (ai-character-scope): honor vae_type="tae". TAE encodes to raw
        # (unnormalized) Wan latents, which is what StreamDiffusionV2 expects, and
        # avoids the full Wan VAE's conv3d encoder that dominates frame time on
        # sm_120. lightvae/lighttae normalize latents, so they still fall back to wan.
        vae_type = getattr(config, "vae_type", "wan")
        vae_type = getattr(vae_type, "value", vae_type)
        if vae_type == "tae":
            vae = create_vae(model_dir=model_dir, vae_type="tae")
            if os.environ.get("SCOPE_COMPILE_TAE") == "1":
                # Local patch: compile TAE streaming encode/decode (Inductor also moves the
                # convs to channels-last); ~+3.5% FPS at 576x320.
                from ..wan2_1.vae import tae as tae_module

                if not getattr(tae_module._apply_model_parallel_streaming, "_scope_compiled", False):
                    compiled = torch.compile(tae_module._apply_model_parallel_streaming, dynamic=False)
                    compiled._scope_compiled = True
                    tae_module._apply_model_parallel_streaming = compiled
        else:
            if vae_type != "wan":
                logger.warning(
                    f"vae_type={vae_type} is not latent-compatible with StreamDiffusionV2; using wan"
                )
                vae_type = "wan"
            vae = StreamDiffusionV2WanVAEWrapper(
                model_dir=model_dir, model_name=base_model_name
            )
        print(f"Loaded VAE (type={vae_type}) in {time.time() - start:.3f}s")
        # Move VAE to target device and use target dtype
        vae = vae.to(device=device, dtype=dtype)

        # Create components config
        if stage_callback:
            stage_callback("Initializing pipeline...")
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("vae", vae)
        components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(
            device=device,
            dtype=dtype,
        )
        components.add("embedding_blender", embedding_blender)

        self.blocks = StreamDiffusionV2Blocks()
        # Local patch (ai-character-scope): load the depth model now (if VACE is configured)
        # instead of on the first depth_vace chunk, which caused a mid-stream CPU spike.
        if os.environ.get("SCOPE_PRELOAD_DEPTH") == "1" and getattr(config, "vace_path", None):
            from ..video_depth_anything.gpu_depth import GpuStreamingDepth

            self.blocks.sub_blocks["depth_vace"]._depth = GpuStreamingDepth(device=device)
        self.components = components
        self.state = PipelineState()
        # These need to be set right now because InputParam.default on the blocks
        # does not work properly
        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", 1.0)
        self.state.set("noise_scale", 0.7)
        self.state.set("noise_controller", True)

        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "base_seed", 42))

        self.first_call = True
        self.last_mode = None  # Track mode for transition detection

    def prepare(self, **kwargs) -> Requirements | None:
        """Return input requirements based on current mode."""
        return prepare_for_mode(self.__class__, self.components.config, kwargs)

    def __call__(self, **kwargs) -> dict:
        self.first_call, self.last_mode = handle_mode_transition(
            self.state, self.components.vae, self.first_call, self.last_mode, kwargs
        )
        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> dict:
        # Handle runtime LoRA scale updates before writing into state.
        lora_scales = kwargs.get("lora_scales")
        if lora_scales is not None:
            self._handle_lora_scale_updates(
                lora_scales=lora_scales, model=self.components.generator.model
            )
            # Trigger cache reset on LoRA scale updates if manage_cache is enabled
            if self.state.get("manage_cache", True):
                kwargs["init_cache"] = True

        # Local patch (ai-character-scope): live attention steering (see attention_steer.py)
        if kwargs.get("attention_weights") is not None:
            attention_steer.set_weights(kwargs["attention_weights"])
        if kwargs.get("sink_frames") is not None:
            attention_steer.set_sink_frames(self.components.generator.model, kwargs["sink_frames"])

        for k, v in kwargs.items():
            self.state.set(k, v)

        # Clear transition from state if not provided to prevent stale transitions
        if "transition" not in kwargs:
            self.state.set("transition", None)

        # Clear video from state if not provided to prevent stale video data
        if "video" not in kwargs:
            self.state.set("video", None)

        # Clear vace_ref_images from state if not provided to prevent encoding on chunks where they weren't sent
        if "vace_ref_images" not in kwargs:
            self.state.set("vace_ref_images", None)

        # Local patch (ai-character-scope): likewise clear VACE input frames/masks, otherwise
        # after a "use input video for VACE" session every later chunk keeps re-encoding the
        # last stale frame as VACE conditioning (slower, and conditions on a frozen frame).
        for key in ("vace_input_frames", "vace_input_masks"):
            if key not in kwargs:
                self.state.set(key, None)

        # Clear extension mode frame images from state if not provided to prevent reuse on non-extension chunks
        if "first_frame_image" not in kwargs:
            self.state.set("first_frame_image", None)
        if "last_frame_image" not in kwargs:
            self.state.set("last_frame_image", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        # Apply mode-specific defaults (noise_scale, noise_controller)
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return {"video": postprocess_chunk(self.state.values["output_video"])}
