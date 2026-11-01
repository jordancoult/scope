"""Local port (ai-character-scope): LongLive 2.0 (NVIDIA, Wan2.2-TI2V-5B, NVFP4) as a Scope text-to-video
streaming pipeline. Model code lives in the NVlabs repo clone (longlive2/LongLive2.0, set up per
tools/longlive2/README.md); this pipeline drives it chunk by chunk and decodes with TAEW2.2."""

from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class LongLive2Config(BasePipelineConfig):
    pipeline_id = "longlive2"
    pipeline_name = "LongLive 2.0"
    pipeline_description = (
        "NVIDIA LongLive 2.0 (Wan2.2-TI2V-5B, 2-step NVFP4) streaming text-to-video. Local port: "
        "1024x576, NVFP4 KV cache, TAEW2.2 decoder. Prompt-driven only (no camera / VACE)."
    )
    estimated_vram_gb = 11.0
    artifacts = []
    inputs = []
    outputs = ["video"]
    supports_cache_management = True
    min_dimension = 16
    modified = True

    height: int = Field(default=576, ge=16, description="Output height (multiple of 16; the model degrades below ~576)",
                        json_schema_extra=ui_field_config(order=1, component="resolution", is_load_param=True))
    width: int = Field(default=1024, ge=16, description="Output width (multiple of 16)",
                       json_schema_extra=ui_field_config(order=1, component="resolution", is_load_param=True))
    base_seed: int = Field(default=42, ge=0, description="Random seed",
                           json_schema_extra=ui_field_config(order=2, is_load_param=True, label="Seed"))
    sampling_steps: int = Field(default=2, ge=1, le=4, description="Denoising steps (the checkpoint is distilled for 2)",
                                json_schema_extra=ui_field_config(order=3, is_load_param=True))
    local_attn_size: int = Field(default=24, ge=8, le=32, description="Memory window in latent frames (stock 32)",
                                 json_schema_extra=ui_field_config(order=4, is_load_param=True))
    kv_quant: bool = Field(default=True, description="NVFP4 KV cache (needed above 640x352 on 16 GB)",
                           json_schema_extra=ui_field_config(order=5, is_load_param=True))
    torch_compile: bool = Field(default=True, description="torch.compile the DiT (+~10%, slow first chunks)",
                                json_schema_extra=ui_field_config(order=6, is_load_param=True))
    stream_latents: int = Field(default=720, ge=64, description="Latent frames per stream before it restarts (4 video frames each)",
                                json_schema_extra=ui_field_config(order=7, is_load_param=True))

    modes = {"text": ModeDefaults(default=True)}
