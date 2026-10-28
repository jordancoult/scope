"""Fused NVFP4 activation quantization in Triton (local patch, ai-character-scope).

One pass over a bf16 [M, K] activation: per 16-element block amax -> e4m3 scale -> round to
e2m1 -> pack two codes per byte, writing the scales directly in cuBLAS's swizzled 128x4 tile
layout (what torch._scaled_mm expects for NVFP4). Replaces a chain of ~6 PyTorch kernels.
"""

import torch
import triton
import triton.language as tl

BLOCKS_PER_PROG = 16  # 16 blocks x 16 elements = 256 columns per program


@triton.jit
def _nvfp4_quant_kernel(x_ptr, q_ptr, s_ptr, K, n_cb, stride_x, BLOCKS: tl.constexpr):
    r = tl.program_id(0)
    g = tl.program_id(1)
    half = tl.arange(0, BLOCKS * 8)
    col_even = g * BLOCKS * 16 + 2 * half
    row_ptr = x_ptr + r * stride_x
    xe = tl.load(row_ptr + col_even, mask=col_even < K, other=0.0).to(tl.float32)
    xo = tl.load(row_ptr + col_even + 1, mask=col_even + 1 < K, other=0.0).to(tl.float32)
    xe2 = tl.reshape(xe, (BLOCKS, 8))
    xo2 = tl.reshape(xo, (BLOCKS, 8))
    amax = tl.maximum(tl.max(tl.abs(xe2), axis=1), tl.max(tl.abs(xo2), axis=1))  # [BLOCKS]
    scale_e4 = (amax / 6.0).to(tl.float8e4nv)
    scale_f = scale_e4.to(tl.float32)
    inv = tl.where(scale_f > 0, 1.0 / scale_f, 0.0)

    ye = xe2 * inv[:, None]
    yo = xo2 * inv[:, None]
    ae = tl.abs(ye)
    ao = tl.abs(yo)
    # round-to-nearest over e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}
    ce = ((ae > 0.25).to(tl.int32) + (ae > 0.75).to(tl.int32) + (ae > 1.25).to(tl.int32)
          + (ae > 1.75).to(tl.int32) + (ae > 2.5).to(tl.int32) + (ae > 3.5).to(tl.int32)
          + (ae > 5.0).to(tl.int32)) | ((ye < 0).to(tl.int32) << 3)
    co = ((ao > 0.25).to(tl.int32) + (ao > 0.75).to(tl.int32) + (ao > 1.25).to(tl.int32)
          + (ao > 1.75).to(tl.int32) + (ao > 2.5).to(tl.int32) + (ao > 3.5).to(tl.int32)
          + (ao > 5.0).to(tl.int32)) | ((yo < 0).to(tl.int32) << 3)
    packed = tl.reshape(ce | (co << 4), (BLOCKS * 8,)).to(tl.uint8)
    pcol = g * BLOCKS * 8 + half
    tl.store(q_ptr + r * (K // 2) + pcol, packed, mask=pcol < K // 2)

    # scale for (row r, block column c) in cuBLAS block-scaled layout
    c = g * BLOCKS + tl.arange(0, BLOCKS)
    rb = r // 128
    r128 = r % 128
    idx = ((rb * n_cb + c // 4) * 32 + (r128 % 32)) * 16 + (r128 // 32) * 4 + (c % 4)
    tl.store(s_ptr + idx, scale_e4, mask=c < K // 16)


def quant_nvfp4_act(x: torch.Tensor):
    """x: [M, K] bf16, K % 32 == 0 -> (fp4 [M, K/2], blocked e4m3 scales padded to 128-row tiles).

    M need not be a multiple of 128: only the (tiny) scale buffer is padded, never the activation.
    """
    m, k = x.shape
    n_cb = (k // 16 + 3) // 4
    n_rb = (m + 127) // 128
    q = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    pad_needed = (k // 16) % 4 or m % 128
    alloc = torch.zeros if pad_needed else torch.empty  # padded scale entries must be 0
    s = alloc(n_rb * n_cb * 128 * 4, dtype=torch.float8_e4m3fn, device=x.device)
    grid = (m, triton.cdiv(k, BLOCKS_PER_PROG * 16))
    _nvfp4_quant_kernel[grid](x, q, s, k, n_cb, x.stride(0), BLOCKS=BLOCKS_PER_PROG)
    return q.view(torch.float4_e2m1fn_x2), s
