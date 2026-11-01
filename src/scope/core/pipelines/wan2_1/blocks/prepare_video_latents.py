import os
from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)


class PrepareVideoLatentsBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("vae_temporal_downsample_factor", 4),
        ]

    @property
    def description(self) -> str:
        return "Prepare Video Latents block that generates noisy latents for a video that will be used for video generation"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video",
                required=True,
                type_hint=list[torch.Tensor] | torch.Tensor,
                description="Input video to convert into noisy latents",
            ),
            InputParam(
                "fixed_noise",
                type_hint=bool,
                default=None,
                description="Reuse the same noise every chunk (local patch; overrides SCOPE_FIXED_NOISE)",
            ),
            InputParam(
                "base_seed",
                type_hint=int,
                default=42,
                description="Base seed for random number generation",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index for current block",
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=0.7,
                description="Amount of noise added to video",
            ),
            # Local patch (ai-character-scope): person hint. Inside person_mask (pixel-res
            # [B, 1, T, H, W], from the depth block) the latents are pulled toward the camera
            # latents by person_hint, elsewhere they stay as noisy as noise_scale says. With
            # noise 1.0 this keeps the person's colours/identity while the scene is invented.
            InputParam("person_mask", default=None, type_hint=torch.Tensor | None),
            InputParam("person_hint", default=0.0, type_hint=float),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Noisy latents to denoise",
            ),
            OutputParam("generator", description="Random number generator"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        # Encode frames to latents using VAE
        # VAE returns [B, F, C, H, W] which is what DenoiseBlock/Generator expect
        latents = components.vae.encode_to_latent(block_state.video)

        # The default param for InputParam does not work right now
        # The workaround is to set the default values here
        base_seed = block_state.base_seed
        if base_seed is None:
            base_seed = 42

        # Create generator from seed for reproducible generation
        # Local patch (ai-character-scope): SCOPE_FIXED_NOISE=1 reuses the same noise every chunk
        # (as StreamDiffusion v1 does) to cut frame-to-frame shimmer from fresh random noise.
        fixed = block_state.fixed_noise
        if fixed is None:
            fixed = os.environ.get("SCOPE_FIXED_NOISE") == "1"
        if fixed:
            block_seed = base_seed
        else:
            block_seed = base_seed + block_state.current_start_frame
        rng = torch.Generator(device=components.config.device).manual_seed(block_seed)

        # Generate empty latents (noise)
        noise = torch.randn(
            latents.shape,
            device=components.config.device,
            dtype=components.config.dtype,
            generator=rng,
        )
        # Determine how noisy the latents should be
        # Higher noise scale -> noiser latents, less of inputs preserved
        # Lower noise scale -> less noisy latents, more of inputs preserved
        block_state.latents = noise * block_state.noise_scale + latents * (
            1 - block_state.noise_scale
        )
        hint = float(block_state.person_hint or 0.0)
        mask = block_state.person_mask
        if hint > 0 and isinstance(mask, torch.Tensor):
            b, f, _, lh, lw = latents.shape
            m = torch.nn.functional.adaptive_avg_pool3d(mask.float(), (f, lh, lw))  # [B, 1, F, h, w]
            m = (m.permute(0, 2, 1, 3, 4) * hint).to(latents.dtype)  # [B, F, 1, h, w]
            block_state.latents = block_state.latents * (1 - m) + latents * m
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state
