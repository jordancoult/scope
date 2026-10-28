"""FP8 row-wise linear layers for consumer Blackwell / Ada (local patch, ai-character-scope).

torchao's Float8DynamicActivationFloat8WeightConfig(PerTensor) was slower than bf16 on an
RTX 5080 Laptop: the per-tensor activation amax runs as separate unfused kernels. This keeps
the weights pre-quantized per output channel (static) and quantizes activations per row with a
single compiled kernel, then calls torch._scaled_mm, which measured ~1.6-1.9x faster than bf16
F.linear for Wan-1.3B shapes (M=720..2880, K/N in {1536, 8960}).

Enable for StreamDiffusionV2 with SCOPE_FP8_ROWWISE=1.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quant_rowwise(x: torch.Tensor):
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / E4M3_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


_quant_rowwise_compiled = torch.compile(_quant_rowwise, dynamic=True)


class FP8RowwiseLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach()
        wq, ws = _quant_rowwise(w.to(torch.bfloat16))
        # _scaled_mm wants the second operand column-major: store W^T as a view of W.
        self.register_buffer("weight_fp8_t", wq.t(), persistent=False)
        self.register_buffer("weight_scale_t", ws.t().contiguous(), persistent=False)
        bias = linear.bias.detach().to(torch.bfloat16) if linear.bias is not None else None
        self.register_buffer("bias", bias, persistent=False)
        self.out_dtype = torch.bfloat16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        m = x2.shape[0]
        pad = -m % 16  # _scaled_mm needs M % 16 == 0 (e.g. 544x304 has 646 tokens/frame)
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        xq, xs = _quant_rowwise_compiled(x2)
        out = torch._scaled_mm(
            xq, self.weight_fp8_t, scale_a=xs, scale_b=self.weight_scale_t,
            bias=self.bias, out_dtype=self.out_dtype,
        )
        if pad:
            out = out[:m]
        return out.reshape(*shape[:-1], self.out_features)


def convert_blocks_to_fp8_rowwise(model: nn.Module, min_features: int = 1024) -> int:
    """Replace nn.Linear layers inside transformer blocks (DiT + VACE) with FP8RowwiseLinear."""
    converted = 0
    for block in [m for m in model.modules() if type(m).__name__.endswith("AttentionBlock")]:
        for name, child in list(block.named_modules()):
            if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= min_features:
                parent = block.get_submodule(name.rsplit(".", 1)[0]) if "." in name else block
                setattr(parent, name.rsplit(".", 1)[-1], FP8RowwiseLinear(child))
                converted += 1
    logger.info(f"fp8 rowwise: converted {converted} linear layers")
    return converted
