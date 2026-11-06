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


from .prepare_latents import forever_noise_seed

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
            # Local patch: output feedback. Start each chunk from a blend of the previous chunk's
            # denoised output and the new camera latents (feedback = share of the previous output),
            # so a 1-step model keeps refining its own result instead of restarting from the camera.
            # feedback_mode "last" repeats the newest output latent frame, "chunk" reuses all of them,
            # "extrapolate" continues the last chunk's motion trend (for driving / walking scenes).
            InputParam("feedback", default=0.0, type_hint=float),
            InputParam("feedback_mode", default="last", type_hint=str),
            # how much of the last chunk's motion to carry forward in "extrapolate" mode (1 = all)
            InputParam("feedback_motion", default=1.0, type_hint=float),
            InputParam("init_cache", default=False, type_hint=bool),
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
            block_seed = forever_noise_seed(state, base_seed, block_state.current_start_frame, latents.shape[1])
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
        base = latents
        fb = float(block_state.feedback or 0.0)
        prev = state.get("latents")  # still the previous chunk's denoised output at this point
        have_prev = (not block_state.init_cache and isinstance(prev, torch.Tensor)
                     and prev.shape[0] == latents.shape[0] and prev.shape[2:] == latents.shape[2:])
        if fb > 0 and not have_prev:
            # Nothing to feed back yet (first chunk / reset): start this chunk from the prompt alone
            # (full noise), otherwise the low feedback noise lets the camera's real room through and
            # feedback then preserves it. The bridge re-sends noise_scale on the next call.
            block_state.noise_scale = 1.0
            state.set("noise_scale", 1.0)
        if fb > 0 and have_prev:
            f = latents.shape[1]
            if block_state.feedback_mode == "chunk" and prev.shape[1] == f:
                prev = prev.to(latents.dtype)
            elif block_state.feedback_mode == "extrapolate" and prev.shape[1] > 1:
                # continue the motion: extend the last chunk's trend (last - first frame) forward
                p = prev.to(torch.float32)
                k = 1.0 if block_state.feedback_motion is None else float(block_state.feedback_motion)
                v = (p[:, -1:] - p[:, :1]) / (p.shape[1] - 1) * k
                steps = torch.arange(1, f + 1, device=p.device, dtype=p.dtype).view(1, f, 1, 1, 1)
                prev = (p[:, -1:] + v * steps).to(latents.dtype)
            else:
                prev = prev[:, -1:].to(latents.dtype).expand(-1, f, -1, -1, -1)
            base = prev * fb + latents * (1 - fb)
        block_state.latents = noise * block_state.noise_scale + base * (
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
