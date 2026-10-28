"""NVFP4 linear layers for consumer Blackwell (sm_120) (local patch, ai-character-scope).

torch 2.9's torch._scaled_mm runs NVFP4 (e2m1 values, one e4m3 scale per 16 elements) on the
RTX 50-series FP4 tensor cores, measured 2.7-3.4x faster than fp8 for Wan-1.3B FFN shapes.
Two-level scaling as in the NVFP4 recipe: a per-tensor fp32 scale keeps the per-block e4m3
scales in e4m3's normal range (tiny weight blocks would otherwise land in subnormals).

Enable for StreamDiffusionV2 with SCOPE_FP4_FFN=1 (FFN layers only; attention projections stay fp8).
"""

import torch
import torch.nn as nn

from .nvfp4_triton import quant_nvfp4_act

BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
# midpoints between e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} -> round-to-nearest codes 0..7
_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _quant_nvfp4(x: torch.Tensor, bounds: torch.Tensor):
    """x [M, K] -> (packed fp4 [M, K/2], block scales e4m3 [M, K/16], global scale fp32 [])."""
    m, k = x.shape
    xf = x.float()
    gscale = (xf.abs().amax() / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-12)
    xb = xf.view(m, k // BLOCK, BLOCK)
    bscale = (xb.abs().amax(-1, keepdim=True) / (E2M1_MAX * gscale)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    xs = xb / (bscale.float() * gscale).clamp(min=1e-30)
    code = torch.bucketize(xs.abs(), bounds).to(torch.uint8) | ((xs < 0).to(torch.uint8) << 3)
    code = code.view(m, k)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, bscale.view(m, k // BLOCK), gscale


def _to_blocked(s: torch.Tensor) -> torch.Tensor:
    """cuBLAS block-scale layout: 128x4 tiles, swizzled into 32x16 (same as torchao.to_blocked)."""
    rows, cols = s.shape
    n_rb, n_cb = -(-rows // 128), -(-cols // 4)
    if rows % 128 or cols % 4:
        s = torch.nn.functional.pad(s.view(torch.uint8), (0, n_cb * 4 - cols, 0, n_rb * 128 - rows)).view(s.dtype)
    blocks = s.view(n_rb, 128, n_cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _quant_act(x: torch.Tensor, bounds: torch.Tensor):
    packed, bscale, gscale = _quant_nvfp4(x, bounds)
    return packed, _to_blocked(bscale), gscale


_quant_act_compiled = torch.compile(_quant_act, dynamic=True)


class FP4Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        assert self.in_features % 32 == 0, "NVFP4 needs K % 32 == 0"
        dev = linear.weight.device
        self.register_buffer("bounds", _BOUNDS.to(dev), persistent=False)
        wq, ws, wg = _quant_nvfp4(linear.weight.detach().to(torch.bfloat16), self.bounds)
        self.register_buffer("weight_fp4", wq.view(torch.float4_e2m1fn_x2), persistent=False)
        self.register_buffer("weight_scale", _to_blocked(ws), persistent=False)
        self.register_buffer("weight_gscale", wg, persistent=False)
        self.weight_gscale_f = float(wg)  # one host sync at load time, none per call
        bias = linear.bias.detach().to(torch.bfloat16) if linear.bias is not None else None
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        # Fused Triton quantizer (one pass); activation block scales are absolute (no global scale)
        xq, xs = quant_nvfp4_act(x2)
        out = torch._scaled_mm(
            xq, self.weight_fp4.t(), scale_a=xs, scale_b=self.weight_scale, out_dtype=torch.bfloat16,
        )
        # weight global scale + bias in one pass
        if self.bias is not None:
            out = torch.add(self.bias, out, alpha=self.weight_gscale_f)
        else:
            out = out * self.weight_gscale_f
        return out.reshape(*shape[:-1], self.out_features)


def convert_ffn_to_fp4(model: nn.Module, skip_first: int = 0, skip_last: int = 0, layers=(0, 2)) -> int:
    """Replace the FFN linears of DiT blocks (not VACE blocks) with FP4Linear."""
    blocks = [m for m in model.modules() if type(m).__name__.endswith("AttentionBlock") and hasattr(m, "ffn")
              and "Vace" not in type(m).__name__]
    converted = 0
    for i, block in enumerate(blocks):
        if i < skip_first or i >= len(blocks) - skip_last:
            continue
        for idx, layer in enumerate(block.ffn):
            if idx not in layers:
                continue
            if isinstance(layer, nn.Linear) or type(layer).__name__ == "FP8RowwiseLinear":
                lin = layer
                if type(layer).__name__ == "FP8RowwiseLinear":  # rebuild a bf16 Linear from fp8
                    lin = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None,
                                    device=layer.weight_fp8_t.device, dtype=torch.bfloat16)
                    with torch.no_grad():
                        lin.weight.copy_((layer.weight_fp8_t.t().float() * layer.weight_scale_t.t()).to(torch.bfloat16))
                        if layer.bias is not None:
                            lin.bias.copy_(layer.bias)
                block.ffn[idx] = FP4Linear(lin)
                converted += 1
    return converted
