"""All-GPU streaming depth for VACE conditioning (local patch, ai-character-scope).

The stock VideoDepthAnythingPipeline round-trips every frame through CPU numpy + cv2 and
normalizes depth per 4-frame chunk (which flickers). On a laptop where the CPU is the
bottleneck that costs more than the model. This keeps frames on the GPU, runs VDA-Small in
fp16 with its temporal cache, and normalizes with a slow running min/max.
"""

import torch
import torch.nn.functional as F

from scope.core.config import get_model_file_path

from .modules import VideoDepthAnything
from .pipeline import MODEL_CONFIG

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class GpuStreamingDepth:
    def __init__(self, device="cuda", input_long_side=504, ema=0.9, frames_per_chunk=1):
        self.device = torch.device(device)
        # VACE's encoder folds each 4-frame chunk into one latent frame, so depth for the newest
        # frame repeated x4 is near-equivalent and 4x cheaper (VDA-S is launch-bound: ~7 ms/call).
        self.frames_per_chunk = frames_per_chunk
        self.input_long_side = input_long_side
        self.ema = ema
        model = VideoDepthAnything(**MODEL_CONFIG, metric=False)
        model.load_state_dict(torch.load(
            get_model_file_path("Video-Depth-Anything-Small/video_depth_anything_vits.pth"),
            map_location="cpu", weights_only=True))
        self.model = model.to(self.device).half().eval()
        self.mean, self.std = _MEAN.to(self.device), _STD.to(self.device)
        self.reset()

    def reset(self):
        self.cache = None
        self.lo = self.hi = None

    def _input_size(self, h, w):
        scale = self.input_long_side / max(h, w)
        return max(14, round(h * scale / 14) * 14), max(14, round(w * scale / 14) * 14)

    @torch.no_grad()
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """video: [B=1, C=3, T, H, W] in [-1, 1]. Returns depth frames, same shape/range (near = bright)."""
        _, _, t_out, h, w = video.shape
        if self.frames_per_chunk < t_out:
            video = video[:, :, -self.frames_per_chunk:]
        t = video.shape[2]
        frames = (video[0].permute(1, 0, 2, 3).float() + 1) / 2  # [T, 3, H, W] in [0, 1]
        ih, iw = self._input_size(h, w)
        x = F.interpolate(frames, size=(ih, iw), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).half()
        depths = []
        for i in range(t):  # streaming mode: one frame at a time with temporal cache
            d, self.cache = self.model(x[i : i + 1].unsqueeze(0), cached_hidden_state_list=self.cache)
            depths.append(d[0])
        d = torch.cat(depths).float().unsqueeze(1)  # [T, 1, ih, iw]
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        lo, hi = torch.quantile(d.flatten(), 0.02), torch.quantile(d.flatten(), 0.98)
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo = self.ema * self.lo + (1 - self.ema) * lo
            self.hi = self.ema * self.hi + (1 - self.ema) * hi
        d = ((d - self.lo) / (self.hi - self.lo).clamp(min=1e-6)).clamp(0, 1)
        if t < t_out:
            d = torch.cat([d[:1].expand(t_out - t, -1, -1, -1), d])  # pad to chunk length
        return (d * 2 - 1).repeat(1, 3, 1, 1).permute(1, 0, 2, 3).unsqueeze(0).to(video.dtype)
