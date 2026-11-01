"""Local port (ai-character-scope): LongLive 2.0 streaming text-to-video inside Scope.

The NVlabs code generates a whole clip in one loop. Here its per-chunk loop (`_inference_inner`) is
turned into a Python generator at load time (one `yield latents` after each chunk is recorded), so each
Scope tick produces exactly one chunk (8 latent frames -> 32 video frames), decoded by a streaming
TAEW2.2. Weights, attention, KV-cache logic are untouched NVlabs code. See
experiments/2026-11-01-longlive2/FINDINGS.md for the settings (1024x576, 2 steps, NVFP4 KV, window 24,
clean pass every 2nd chunk).

Live parameters (per call): prompts, init_cache/reset_cache, clean_kv_every (1|2; skip_clean_kv -> never),
attention_backend ("default" = exact flash-attention here, "sage2first", "sage2"/"flash").
"""

import inspect
import logging
import os
import sys
import textwrap
import time
import types
import warnings
from pathlib import Path

import torch

from ..interface import Pipeline, Requirements
from .schema import LongLive2Config

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("AICS_ROOT", Path(__file__).resolve().parents[6]))
REPO = Path(os.environ.get("LL2_REPO", ROOT / "longlive2" / "LongLive2.0"))
MODELS = Path(os.environ.get("LL2_MODELS", ROOT / "models" / "LongLive-2.0-5B"))


class LongLive2Pipeline(Pipeline):
    @classmethod
    def get_config_class(cls):
        return LongLive2Config

    def __init__(self, height=576, width=1024, base_seed=42, sampling_steps=2, local_attn_size=24,
                 kv_quant=True, torch_compile=True, stream_latents=720, **_):
        super().__init__()
        t0 = time.time()
        # The DiT worker thread runs a lot of Python per chunk and holds the GIL most of the time; the WebRTC
        # event loop (RTP packetize/encrypt/send) then waits up to the 5 ms switch interval at every GIL handoff
        # and the output stalled at 2-4 fps. A shorter interval lets it in.
        sys.setswitchinterval(float(os.environ.get("LL2_GIL_SWITCH_S", "0.0005")))
        self.height, self.width, self.seed = int(height) // 16 * 16, int(width) // 16 * 16, int(base_seed)
        self.stream_latents = max(64, int(stream_latents) // 8 * 8)
        self.dev = torch.device("cuda")
        if not REPO.is_dir():
            raise RuntimeError(f"LongLive 2.0 repo not found at {REPO} (see tools/longlive2/README.md)")
        # The NVlabs code uses relative paths (wan_models/...) and generic top-level module names.
        link = Path.cwd() / "wan_models"
        if not link.exists():
            link.symlink_to(REPO / "wan_models")
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        deps = str(REPO.parent / ".deps")
        if deps not in sys.path:
            sys.path.append(deps)  # fouroversix, sentencepiece (appended: never shadow Scope's own packages)
        warnings.filterwarnings("ignore", module="fouroversix")
        import logging as _lg
        torch._logging.set_logs(dynamo=_lg.ERROR, inductor=_lg.ERROR)

        from omegaconf import OmegaConf

        import wan_5b.modules.causal_model as cm
        import utils.wan_5b_wrapper as ww
        from wan_5b.modules.t5 import umt5_xxl
        from wan_5b.modules.tokenizers import HuggingfaceTokenizer

        # Base Wan2.2 diffusion weights aren't downloaded: build from config.json; the checkpoint has all weights.
        cm.CausalWanModel.from_pretrained = classmethod(
            lambda cls, path, **kw: cls.from_config(cls.load_config(path), **kw))

        # Self-attention backend switch (exact flash-attention by default for this model). The mode lives on
        # the module (not a closure over self), so an unloaded pipeline can actually be freed.
        cm.LL2_ATTN_MODE = "default"
        if not getattr(cm, "_aics_attn_patched", False):
            _fa = cm.attention
            from sageattention import sageattn  # SageAttention 2.2 (installed in Scope's environment)

            def _attention(q, k, v, *a, **kw):
                m = cm.LL2_ATTN_MODE
                if m == "sage2" or (m == "sage2first" and getattr(cm, "LL2_PASS_STEP", -1) == 0):
                    return sageattn(q.contiguous(), k.contiguous(), v.contiguous(), tensor_layout="NHD", is_causal=False)
                return _fa(q, k, v, *a, **kw)

            cm.attention = _attention
            cm._aics_attn_patched = True
        self._cm = cm

        from pipeline import CausalDiffusionInferencePipeline as CDIP
        from utils.config import normalize_config
        from utils.inference_utils import setup_nvfp4_pipeline

        cfg = OmegaConf.load(str(REPO / "configs" / "nvfp4" / "inference_nvfp4.yaml"))
        cfg.checkpoints.generator_ckpt = str(MODELS / "model_4o6.pt")
        cfg.model_quant_use_transformer_engine = False
        cfg.model_quant_te_fallback_to_fouroversix = False
        cfg.inference.kv_quant = bool(kv_quant)
        cfg.model_kwargs.local_attn_size = int(local_attn_size)
        cfg.inference.vae_device = "cuda:0"
        cfg.inference.streaming_vae = False
        cfg.inference.sampling_steps = int(sampling_steps)
        cfg.torch_compile = False
        cfg.num_output_frames = self.stream_latents
        cfg.data.image_or_video_shape = [1, self.stream_latents, 48, self.height // 16, self.width // 16]
        config = normalize_config(cfg)
        torch.set_grad_enabled(False)

        class CpuT5(torch.nn.Module):
            """WanTextEncoder in bf16 on the CPU (the stock one moves a float32 T5, 22 GB, to the GPU)."""
            def __init__(self):
                super().__init__()
                self.text_encoder = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16,
                                             device=torch.device("cpu")).eval().requires_grad_(False)
                self.text_encoder.load_state_dict(torch.load(
                    "wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu", weights_only=False))
                self.tokenizer = HuggingfaceTokenizer(name="wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/", seq_len=512,
                                                      clean="whitespace")

            device = property(lambda self: torch.device("cpu"))
            forward = ww.WanTextEncoder.forward

        dev = self.dev

        class CachedText(torch.nn.Module):
            """Caches one GPU embedding per unique prompt (the pipeline encodes one copy per chunk)."""
            def __init__(self, enc):
                super().__init__()
                self._hidden = [enc]
                self.cache = {}

            def to(self, *a, **k):
                return self

            def forward(self, text_prompts):
                for t in set(text_prompts):
                    if t not in self.cache:
                        self.cache[t] = self._hidden[0](text_prompts=[t])["prompt_embeds"].to(dev)
                return {"prompt_embeds": torch.cat([self.cache[t] for t in text_prompts], 0)}

        pipe = CDIP(config, device=dev, text_encoder=CpuT5())
        pipe.text_encoder = CachedText(pipe.text_encoder)
        setup_nvfp4_pipeline(pipe, config, dev)
        pipe.generator.model.eval().requires_grad_(False)
        if torch_compile:
            pipe.generator.configure_torch_compile(mode="default", dynamic=False, suppress_errors=True)

        # Turn the per-chunk loop into a generator: yield each chunk's latents right after it is recorded.
        src = textwrap.dedent(inspect.getsource(CDIP._inference_inner))
        marker = "output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents\n"
        if src.count(marker) != 1:
            raise RuntimeError("LongLive 2.0 code changed: chunk marker not found")
        line = next(ln for ln in src.splitlines(True) if ln.strip() == marker.strip())
        indent = line[: len(line) - len(line.lstrip())]
        src = src.replace(marker, marker + indent + "yield latents\n")
        mod = sys.modules[CDIP.__module__]
        ns = {}
        exec(compile(src, inspect.getsourcefile(CDIP), "exec"), mod.__dict__, ns)
        pipe._inference_inner = types.MethodType(ns["_inference_inner"], pipe)
        self.pipe = pipe

        sys.path.insert(0, str(MODELS / "fastvae"))
        from taehv import TAEHV, StreamingTAEHV

        self.tae = StreamingTAEHV(TAEHV(str(MODELS / "fastvae" / "taew2_2.pth")).to(dev, torch.float16))
        self.gen = None
        self.prompt = None
        self.n_streams = 0
        logger.info(f"LongLive 2.0 loaded in {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        print(f"[longlive2] loaded in {time.time() - t0:.0f}s ({self.width}x{self.height}, {sampling_steps} steps, "
              f"window {local_attn_size}, kv_quant={kv_quant}, compile={torch_compile})", flush=True)

    # The NVlabs inference() restores per-module attention settings in a finally block as soon as it returns
    # our generator, so re-apply them before the first chunk runs.
    def _apply_model_settings(self):
        p = self.pipe
        dit = p._dit_model
        dit.local_attn_size = p.local_attn_size
        p._set_all_modules_max_attention_size(p.local_attn_size)
        if p.sink_size is not None:
            p._set_all_modules_sink_size(p.sink_size)
        p._set_all_modules_global_sink_size(p.global_sink_size)
        if p.inference_t_scale is not None:
            dit.t_scale = p.inference_t_scale
        if p._rope_method_override is not None:
            dit.rope_method = p._rope_method_override
        if p._original_seq_len_override is not None:
            dit.original_seq_len = p._original_seq_len_override
        dit.use_relative_rope = p.use_relative_rope

    def _start(self, prompt: str):
        n = self.stream_latents
        g = torch.Generator(device=self.dev).manual_seed(self.seed + self.n_streams)
        noise = torch.randn([1, n, 48, self.height // 16, self.width // 16], device=self.dev, dtype=torch.bfloat16,
                            generator=g)
        self.gen = self.pipe.inference(noise=noise, text_prompts=[[prompt] * (n // 8)], return_latents=True)
        self._apply_model_settings()
        self.tae.reset()
        self.prompt = prompt
        self.n_streams += 1
        print(f"[longlive2] stream {self.n_streams}: {prompt[:80]!r}", flush=True)

    def prepare(self, **kwargs) -> Requirements | None:
        return None  # text-to-video: never waits for input frames

    def __call__(self, **kwargs) -> dict:
        prompts = kwargs.get("prompts") or []
        prompt = (prompts[0].get("text") if prompts and isinstance(prompts[0], dict) else None) or self.prompt or \
            "a cinematic shot"
        if "attention_backend" in kwargs:
            m = str(kwargs["attention_backend"] or "default")
            self._cm.LL2_ATTN_MODE = m if m in ("default", "flash", "sage2", "sage2first") else "default"
        every = 0 if kwargs.get("skip_clean_kv") else int(kwargs.get("clean_kv_every") or 2)
        os.environ["LL2_SKIP_CLEAN"] = "1" if every == 0 else "0"
        os.environ["LL2_CLEAN_EVERY"] = str(max(1, every))
        if self.gen is None or prompt != self.prompt or kwargs.get("init_cache") or kwargs.get("reset_cache"):
            self._start(prompt)
        t0 = time.perf_counter()
        try:
            lat = next(self.gen)
        except StopIteration:
            self._start(prompt)
            lat = next(self.gen)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        frames = []
        f = self.tae.decode(lat.half())
        while f is not None:
            frames.append(f)
            f = self.tae.decode()
        if not frames:
            return {}
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self._t = getattr(self, "_t", [0, 0.0, 0.0])
        self._t[0] += 1; self._t[1] += t1 - t0; self._t[2] += t2 - t1
        if self._t[0] % 10 == 0:
            n = self._t[0]
            print(f"[longlive2] chunk avg: generate {1000 * self._t[1] / n:.0f} ms, decode {1000 * self._t[2] / n:.0f} ms "
                  f"({len(frames)} frames/chunk, {torch.cuda.memory_allocated() / 1e9:.1f} GB)", flush=True)
            self._t = [0, 0.0, 0.0]
        video = torch.cat(frames, 1)[0]  # T C H W in [0, 1]
        # Hand Scope CPU uint8 frames: one copy per chunk. Leaving them on the GPU made the output track do a
        # per-frame GPU->CPU copy that waits behind the next chunk's queued DiT work (~330 ms per frame).
        video = (video.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        return {"video": video}
