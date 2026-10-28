"""Depth-to-VACE block (local patch, ai-character-scope).

When the runtime parameter ``depth_vace`` is true and a video chunk is present, converts the
preprocessed RGB chunk into GPU depth maps and feeds them as ``vace_input_frames``. The RGB
chunk still initializes the latents, so the result keeps the camera framing (latent init)
while VACE depth hints hold the body structure and ``vace_ref_images`` supply the character.
"""

from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam


class DepthVaceBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    def __init__(self):
        super().__init__()
        self._depth = None

    @property
    def description(self) -> str:
        return "Converts the input video chunk to depth maps for VACE structure conditioning"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("video", default=None, type_hint=torch.Tensor | None),
            InputParam("vace_input_frames", default=None, type_hint=torch.Tensor | None),
            InputParam("depth_vace", default=False, type_hint=bool),
            InputParam("init_cache", default=False, type_hint=bool),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("vace_input_frames", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        video = block_state.video
        if block_state.depth_vace and isinstance(video, torch.Tensor):
            if self._depth is None:
                from ..video_depth_anything.gpu_depth import GpuStreamingDepth

                self._depth = GpuStreamingDepth(device=video.device)
            if block_state.init_cache:
                self._depth.reset()
            block_state.vace_input_frames = self._depth(video)
        self.set_block_state(state, block_state)
        return components, state
