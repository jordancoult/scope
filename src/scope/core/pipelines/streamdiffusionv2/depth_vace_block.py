"""Depth-to-VACE block (local patch, ai-character-scope).

When the runtime parameter ``depth_vace`` is true and a video chunk is present, converts the
preprocessed RGB chunk into GPU depth maps and feeds them as ``vace_input_frames``. The RGB
chunk still initializes the latents, so the result keeps the camera framing (latent init)
while VACE depth hints hold the body structure and ``vace_ref_images`` supply the character.
"""

from typing import Any

import torch
import torch.nn.functional as F
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
            # Body-only depth: depth below this (normalized 0..1, 1 = nearest) becomes neutral gray,
            # VACE's "no information" value, so the model invents the background. 0 = off.
            InputParam("depth_cutoff", default=0.0, type_hint=float),
            # Adaptive body-only depth: keep this fraction of the nearest pixels (e.g. 0.35 for a
            # person at a desk) whatever the room depth is. Overrides depth_cutoff when > 0.
            InputParam("depth_keep", default=0.0, type_hint=float),
            # Scene placement: shrink the person (depth map and camera image) to place_scale of the
            # frame, centred at (place_x, place_y) in 0..1. The rest is neutral gray, so the model
            # builds the scene around a smaller, farther-away subject. 1.0 = off.
            InputParam("place_scale", default=1.0, type_hint=float),
            InputParam("place_x", default=0.5, type_hint=float),
            InputParam("place_y", default=0.6, type_hint=float),
            InputParam("person_mask", default=None, type_hint=torch.Tensor | None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("vace_input_frames", type_hint=torch.Tensor), OutputParam("video", type_hint=torch.Tensor),
                OutputParam("person_mask", type_hint=torch.Tensor)]

    @staticmethod
    def _place(x: torch.Tensor, scale: float, cx: float, cy: float, fill: float = 0.0) -> torch.Tensor:
        """[B, C, T, H, W] in -1..1 -> same shape, content shrunk into a gray (0) canvas."""
        b, c, t, h, w = x.shape
        nh, nw = max(2, round(h * scale)), max(2, round(w * scale))
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        small = F.interpolate(frames, size=(nh, nw), mode="area")
        out = torch.full_like(frames, fill)
        y0 = min(max(0, round(cy * h - nh / 2)), h - nh)
        x0 = min(max(0, round(cx * w - nw / 2)), w - nw)
        out[:, :, y0 : y0 + nh, x0 : x0 + nw] = small
        return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(x.dtype)

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        video = block_state.video
        block_state.person_mask = None  # recomputed below when depth runs (never stale)
        if block_state.depth_vace and isinstance(video, torch.Tensor):
            if self._depth is None:
                from ..video_depth_anything.gpu_depth import GpuStreamingDepth

                self._depth = GpuStreamingDepth(device=video.device)
            if block_state.init_cache:
                self._depth.reset()
            frames = self._depth(video)
            c = float(block_state.depth_cutoff or 0.0)
            d = (frames[:, :1] + 1) / 2  # [B, 1, T, H, W] in 0..1, 1 = nearest
            keep = float(block_state.depth_keep or 0.0)
            if 0 < keep < 1:
                c = float(torch.quantile(d.float().flatten()[::7], 1 - keep)) + 0.06  # edge sits at the quantile
            # person mask for the latent hint: the body-only cutoff if set, else the nearest ~40%
            w = ((d - ((c or 0.45) - 0.06)) / 0.12).clamp(0, 1)
            if c > 0:
                frames = frames * w  # background -> 0 (gray = unknown to VACE)
            block_state.vace_input_frames = frames
            block_state.person_mask = w
        scale = float(block_state.place_scale or 1.0)
        if 0 < scale < 0.999 and isinstance(video, torch.Tensor):
            cx, cy = float(block_state.place_x), float(block_state.place_y)
            if block_state.depth_vace and isinstance(block_state.vace_input_frames, torch.Tensor):
                # (Tried pushing the placed depth into the far range so the subject reads as distant:
                # it made the model draw close-ups instead. Neutral gray surroundings work better.)
                block_state.vace_input_frames = self._place(block_state.vace_input_frames, scale, cx, cy)
            if isinstance(block_state.person_mask, torch.Tensor):
                block_state.person_mask = self._place(block_state.person_mask, scale, cx, cy)
            block_state.video = self._place(video, scale, cx, cy)
        self.set_block_state(state, block_state)
        return components, state
