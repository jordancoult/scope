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



FUSED_KV = os.environ.get("SCOPE_FUSED_KV") == "1"
NOISE_T_OFFSET = int(os.environ.get("SCOPE_NOISE_T_OFFSET", "100"))

class DenoiseBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
            ComponentSpec("scheduler", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("vae_spatial_downsample_factor", 8),
        ]

    @property
    def description(self) -> str:
        return "Denoise block that performs iterative denoising"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "height",
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Width of the video",
            ),
            InputParam(
                "kv_cache_attention_bias",
                default=1.0,
                type_hint=float,
                description="Controls how much to rely on past frames in the cache during generation",
            ),
            # The following should be converted to intermediate inputs to denote that they can come from other blocks
            # and can be modified since they are also listed under intermediate outputs. They are included as inputs for now
            # because of what seems to be a bug where intermediate inputs cannot be simplify accessed in block state via
            # block_state.<intermediate_input>
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Noisy latents to denoise",
            ),
            InputParam(
                "current_denoising_step_list",
                required=True,
                type_hint=torch.Tensor,
                description="Current list of denoising steps",
            ),
            InputParam(
                "conditioning_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Conditioning embeddings used to condition denoising",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index for current block",
            ),
            InputParam(
                "start_frame",
                type_hint=int | None,
                description="Starting frame index that overrides current_start_frame",
            ),
            InputParam(
                "kv_cache",
                required=True,
                type_hint=list[dict],
                description="Initialized KV cache",
            ),
            InputParam(
                "crossattn_cache",
                required=True,
                type_hint=list[dict],
                description="Initialized cross-attention cache",
            ),
            InputParam(
                "kv_bank",
                type_hint=list[dict],
                description="Initialized KV memory bank",
            ),
            InputParam(
                "generator",
                required=True,
                description="Random number generator",
            ),
            InputParam(
                "noise_scale",
                type_hint=float | None,
                description="Amount of noise added to video",
            ),
            InputParam(
                "vace_context",
                default=None,
                type_hint=torch.Tensor | None,
                description="VACE context that provides visual conditioning",
            ),
            InputParam(
                "vace_context_scale",
                default=1.0,
                type_hint=float,
                description="Scaling factor for VACE hint injection",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        noise = block_state.latents
        batch_size = noise.shape[0]
        num_frames = noise.shape[1]
        denoising_step_list = block_state.current_denoising_step_list.clone()

        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}

        start_frame = block_state.current_start_frame
        if block_state.start_frame is not None:
            start_frame = block_state.start_frame

        end_frame = start_frame + num_frames

        # Local patch (ai-character-scope): fused clean-KV refresh. Instead of a separate
        # CleanKVCacheBlock forward over this chunk's denoised latents, they are carried into the
        # *next* chunk's first denoising forward as a t=0 prefix (one DiT pass per chunk instead
        # of two; self-attention keeps the prefix causal). Only when the cache provably ends
        # exactly at the previous chunk.
        fused_prefix = None
        if FUSED_KV:
            prev = getattr(self, "_fused_prev", None)
            self._fused_prev = None
            if prev is not None:
                prev_x0, prev_start = prev
                from scope.core.pipelines.streamdiffusionv2.modules.causal_model import _kv_index

                cache_end = _kv_index(block_state.kv_cache[0], "global_end_index")
                if (
                    prev_start + prev_x0.shape[1] == start_frame
                    and cache_end == start_frame * frame_seq_length
                    and prev_x0.shape[0] == batch_size
                    and prev_x0.shape[2:] == noise.shape[2:]
                ):
                    fused_prefix = prev_x0

        if block_state.noise_scale is not None:
            # Higher noise scale -> more denoising steps, more intense changes to input
            # Lower noise scale -> less denoising steps, less intense changes to input
            # Local patch (ai-character-scope): the -100 means the model is told less noise than was
            # mixed in. StreamDiffusionV2 is tuned for that; other models leave the rest as residual
            # noise. SCOPE_NOISE_T_OFFSET overrides it (0 = tell the true noise level).
            denoising_step_list[0] = int(1000 * block_state.noise_scale) - NOISE_T_OFFSET

        # Denoising loop
        for index, current_timestep in enumerate(denoising_step_list):
            if index == 0:
                q_bank = True
            else:
                q_bank = False

            timestep = (
                torch.ones(
                    [batch_size, num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if index == 0 and fused_prefix is not None:
                from scope.core.pipelines.streamdiffusionv2.modules import causal_model

                pf = fused_prefix.shape[1]
                causal_model.FUSED_PREFIX_TOKENS = pf * frame_seq_length
                try:
                    _, fused_pred = components.generator(
                        noisy_image_or_video=torch.cat([fused_prefix, noise], dim=1),
                        conditional_dict=conditional_dict,
                        timestep=torch.cat([torch.zeros_like(timestep[:, :1]).expand(-1, pf), timestep], dim=1),
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        kv_bank=block_state.kv_bank,
                        update_bank=False,
                        q_bank=q_bank,
                        current_start=(start_frame - pf) * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        vace_context=block_state.vace_context,
                        vace_context_scale=block_state.vace_context_scale,
                    )
                finally:
                    causal_model.FUSED_PREFIX_TOKENS = 0
                denoised_pred = fused_pred[:, pf:]
                if index < len(denoising_step_list) - 1:
                    next_timestep = denoising_step_list[index + 1]
                    flattened_pred = denoised_pred.flatten(0, 1)
                    random_noise = torch.randn(
                        flattened_pred.shape,
                        device=flattened_pred.device,
                        dtype=flattened_pred.dtype,
                        generator=block_state.generator,
                    )
                    noise = components.scheduler.add_noise(
                        flattened_pred,
                        random_noise,
                        next_timestep
                        * torch.ones(
                            [batch_size * num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
            elif index < len(denoising_step_list) - 1:
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    kv_bank=block_state.kv_bank,
                    update_bank=False,
                    q_bank=q_bank,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                    vace_context=block_state.vace_context,
                    vace_context_scale=block_state.vace_context_scale,
                )

                next_timestep = denoising_step_list[index + 1]
                # Create noise with same shape and properties as denoised_pred
                flattened_pred = denoised_pred.flatten(0, 1)
                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=block_state.generator,
                )
                noise = components.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones(
                        [batch_size * num_frames],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    kv_bank=block_state.kv_bank,
                    update_bank=False,
                    q_bank=q_bank,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                    vace_context=block_state.vace_context,
                    vace_context_scale=block_state.vace_context_scale,
                )

        block_state.latents = denoised_pred
        if FUSED_KV:
            self._fused_prev = (denoised_pred.detach(), start_frame)

        self.set_block_state(state, block_state)
        return components, state
