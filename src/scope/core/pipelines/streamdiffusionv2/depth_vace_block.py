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
        self._seg = None

    @torch.no_grad()
    def _segment(self, video: torch.Tensor, placed: bool, fade_bottom: float = 0.12) -> torch.Tensor:
        """[B, 3, T, H, W] in -1..1 -> soft person mask [B, 1, T, H, W] in 0..1."""
        if self._seg is None:
            from torchvision.models.segmentation import (
                DeepLabV3_MobileNet_V3_Large_Weights,
                deeplabv3_mobilenet_v3_large,
            )

            w = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self._seg = deeplabv3_mobilenet_v3_large(weights=w).to(video.device).eval().half()
        b, c, t, h, w = video.shape
        step = 3 if t >= 6 else 1  # segment every 3rd frame; the person moves little in 0.1 s
        idx = torch.arange(0, t, step, device=video.device)
        x = video[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).float()
        x = (x + 1) / 2
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        p = self._seg(((x - mean) / std).half())["out"].float().softmax(1)[:, 15:16]  # VOC "person"
        p = F.avg_pool2d(p, 9, stride=1, padding=4)  # feather the outline
        p = p.reshape(b, len(idx), 1, h, w).permute(0, 2, 1, 3, 4)
        p = p.repeat_interleave(step, dim=2)[:, :, :t]
        if placed:  # fade to zero over the outer ~12% so nothing touches the placed rectangle's edge
            ys = torch.linspace(0, 1, h, device=p.device)
            xs = torch.linspace(0, 1, w, device=p.device)
            fy = torch.minimum((ys / 0.12).clamp(0, 1), ((1 - ys) / max(fade_bottom, 1e-3)).clamp(0, 1))
            fx = ((torch.minimum(xs, 1 - xs)) / 0.12).clamp(0, 1)
            win = (fy[:, None] * fx[None, :]) ** 0.5
            p = p * win
        return p

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
            # Cut out the person with a segmentation model (DeepLabV3-MobileNet, ~2 ms/frame)
            # instead of a depth threshold: no room patches, feathered outline, and when placed
            # everything fades out before the edge of the placed rectangle (no visible box).
            InputParam("person_seg", default=False, type_hint=bool),
            # bottom fade of the placed cut-out (0 = the torso runs to the bottom edge, like a body
            # continuing below the frame; fading it turns the person into a floating bust)
            InputParam("place_fade_bottom", default=0.12, type_hint=float),
            # virtual ground: when placed, the surroundings get a smooth full-frame depth gradient
            # (far at the top, nearer at the bottom) and the person sits in it at mid distance, which
            # gives the model a sense of scale without any edges
            InputParam("place_ground", default=False, type_hint=bool),
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
            if block_state.person_seg:
                fb = block_state.place_fade_bottom
                seg = self._segment(video, float(block_state.place_scale or 1.0) < 0.999, 0.12 if fb is None else float(fb))
                if seg.shape[2] != frames.shape[2]:  # depth pads short chunks at the front
                    seg = torch.cat([seg[:, :, :1].expand(-1, -1, frames.shape[2] - seg.shape[2], -1, -1), seg], 2)
                w = seg.to(frames.dtype)
                frames = frames * w  # everything but the person -> 0 (gray = unknown to VACE)
            elif c > 0:
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
            if block_state.place_ground and block_state.depth_vace and isinstance(block_state.person_mask, torch.Tensor):
                vf = block_state.vace_input_frames
                pm = block_state.person_mask.to(vf.dtype)
                if pm.shape[2] != vf.shape[2]:
                    pm = torch.cat([pm[:, :, :1].expand(-1, -1, vf.shape[2] - pm.shape[2], -1, -1), pm], 2)
                h = vf.shape[-2]
                grad = torch.linspace(0.05, 0.5, h, device=vf.device, dtype=vf.dtype).view(1, 1, 1, h, 1)
                person = 0.5 + 0.25 * (vf + 1) / 2
                block_state.vace_input_frames = (grad * (1 - pm) + person * pm) * 2 - 1
            block_state.video = self._place(video, scale, cx, cy)
        self.set_block_state(state, block_state)
        return components, state
