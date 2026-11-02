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


class CleanKVCacheBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("vae_spatial_downsample_factor", 8),
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("record_interval", 3),
        ]

    @property
    def description(self) -> str:
        return "Clean KV Cache block runs the generator with timestep zero and denoised latents to clean the KV cache"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index of current block",
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
                "height", required=True, type_hint=int, description="Height of video"
            ),
            InputParam(
                "width", required=True, type_hint=int, description="Width of video"
            ),
            # Local patch (ai-character-scope): skip the extra clean-memory pass when denoising used 2+
            # steps; the last step's nearly-clean input is kept as the memory instead. Off by default.
            InputParam("skip_clean_kv", default=False, type_hint=bool),
            # run the clean pass only on every Nth chunk (1 = every chunk, stock); 2+ steps only
            InputParam("clean_kv_every", default=1, type_hint=int),
            InputParam(
                "conditioning_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Conditioning embeddings to condition denoising",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        # Local patch (ai-character-scope): with SCOPE_FUSED_KV=1 the clean-KV refresh happens
        # inside the next chunk's denoising forward (see DenoiseBlock), so skip it here.
        # SCOPE_SKIP_CLEAN_KV=1 skips it entirely (faster, visibly worse; experiment only).
        # The first chunk of a video stream still gets a real pass: Reward Forcing's next chunk
        # overlaps it by a frame, so it can't be fused (one extra pass per stream start).
        first_video_chunk = state.get("video") is not None and block_state.current_start_frame == 0
        if os.environ.get("SCOPE_SKIP_CLEAN_KV") == "1" or (
            os.environ.get("SCOPE_FUSED_KV") == "1" and not first_video_chunk
        ):
            self.set_block_state(state, block_state)
            return components, state

        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        record_interval = getattr(components.config, "record_interval", None)
        if (
            record_interval is not None
            and block_state.current_start_frame % (3 * record_interval) == 0
        ):
            update_bank = True
        else:
            update_bank = False

        generator_param = next(components.generator.parameters())

        every = int(block_state.clean_kv_every or 1)
        chunk = block_state.current_start_frame // max(1, block_state.latents.shape[1])
        skip = block_state.skip_clean_kv or (every > 1 and chunk % every != 0)
        if skip and len(state.get("denoising_step_list") or []) >= 2 and not update_bank:
            # (MemFlow still runs the pass on chunks that write its memory bank)
            self.set_block_state(state, block_state)
            return components, state

        _, num_frames, _, _, _ = block_state.latents.shape
        current_end_frame = block_state.current_start_frame + num_frames

        # This is defined to give us timestep = 0 while matching shape expected by the generator.
        # After denoising the KV cache will contain keys/values computed from the noisy input at the final timestep.
        # We want to update the generator with the key/values computed from the final "clean" latent (no noise) which
        # corresponds with timestep = 0.
        # The multiplication by 0 gives us timestep = 0 and is included to illustrate that we could also multiply by
        # a different value (typically a context_noise param).
        context_timestep = (
            torch.ones(
                [1, num_frames],
                device=generator_param.device,
                dtype=generator_param.dtype,
            )
            * 0
        )

        # Run the generator with the clean latent at timestep = 0 to update the KV cache
        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}
        components.generator(
            noisy_image_or_video=block_state.latents,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=block_state.kv_cache,
            crossattn_cache=block_state.crossattn_cache,
            kv_bank=block_state.kv_bank,
            update_bank=update_bank,
            q_bank=False,
            update_cache=True,
            current_start=block_state.current_start_frame * frame_seq_length,
            current_end=current_end_frame * frame_seq_length,
        )

        self.set_block_state(state, block_state)
        return components, state
