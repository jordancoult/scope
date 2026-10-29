"""Local patch (ai-character-scope): live attention steering over the KV cache.

Each query's self-attention over the cache is split into three key regions:
  sink    - the first frames after a reset (the long-term anchor)
  recent  - the rolling window of previous frames
  current - the chunk being generated (its own camera-initialised tokens)
A weight w per region adds log(w) to that region's attention logits, i.e. multiplies its share
of attention by w before renormalising (w < 1 dampens, w > 1 emphasises). With all weights at 1
the stock attention kernel runs untouched, so default output is bit-identical.

Also "text": multiplies the cross-attention (prompt) output added to each block (1 = stock).

Set at runtime with the pipeline parameter
attention_weights={"sink": .., "recent": .., "current": .., "text": ..}.
"""

import torch
import torch.nn.functional as F

from .attention import attention

_weights = {"sink": 1.0, "recent": 1.0, "current": 1.0, "text": 1.0}
_log_w = None  # [3] float32 CUDA tensor; a tensor so torch.compile doesn't recompile per value
_text_w = None  # [1] float32 CUDA tensor
ACTIVE = False
TEXT_ACTIVE = False


def set_weights(weights: dict | None) -> None:
    global _log_w, _text_w, ACTIVE, TEXT_ACTIVE
    if not weights:
        return
    for key in _weights:
        if key in weights and weights[key] is not None:
            _weights[key] = min(4.0, max(0.0 if key == "text" else 0.05, float(weights[key])))
    vals = torch.tensor([_weights["sink"], _weights["recent"], _weights["current"]]).log()
    if _log_w is None:
        _log_w = vals.cuda()
    else:
        _log_w.copy_(vals)
    if _text_w is None:
        _text_w = torch.tensor([_weights["text"]]).cuda()
    else:
        _text_w.fill_(_weights["text"])
    ACTIVE = any(_weights[k] != 1.0 for k in ("sink", "recent", "current"))
    TEXT_ACTIVE = _weights["text"] != 1.0


def scale_cross(y):
    """Cross-attention (prompt) output, scaled by the "text" weight when it isn't 1."""
    return y * _text_w.to(y.dtype) if TEXT_ACTIVE else y


def steered_attention(q, k, v, sink_tokens: int):
    """q: [B, Lq, H, D]; k, v: [B, Lk, H, D] laid out as [sink | recent | current(last Lq)]."""
    if not ACTIVE:
        return attention(q, k, v)
    lq, lk = q.shape[1], k.shape[1]
    sink_end = min(sink_tokens, lk)
    cur_start = max(lk - lq, sink_end)
    idx = torch.ones(lk, dtype=torch.long, device=q.device)
    idx[:sink_end] = 0
    idx[cur_start:] = 2
    bias = _log_w[idx].to(q.dtype).view(1, 1, 1, lk).expand(q.shape[0], q.shape[2], lq, lk)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias
    )
    return out.transpose(1, 2).contiguous()


def set_sink_frames(model, frames: int) -> None:
    """Number of latent frames kept as the never-evicted sink (takes effect after a cache reset)."""
    if getattr(model, "_steer_sink_frames", None) == int(frames):
        return
    model._steer_sink_frames = int(frames)
    for m in model.modules():
        if hasattr(m, "sink_size") and isinstance(m.sink_size, int):
            m.sink_size = int(frames)
