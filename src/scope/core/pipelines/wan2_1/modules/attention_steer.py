"""Local patch (ai-character-scope): live attention steering over the KV cache.

Each query's self-attention over the cache is split into three key regions:
  sink    - the first frames after a reset (the long-term anchor)
  recent  - the rolling window of previous frames
  current - the chunk being generated (its own camera-initialised tokens)
A weight w per region adds log(w) to that region's attention logits, i.e. multiplies its share
of attention by w before renormalising (w < 1 dampens, w > 1 emphasises). With all weights at 1
the stock attention kernel runs untouched, so default output is bit-identical.

Set at runtime with the pipeline parameter attention_weights={"sink": .., "recent": .., "current": ..}.
"""

import torch
import torch.nn.functional as F

from .attention import attention

_weights = {"sink": 1.0, "recent": 1.0, "current": 1.0}
_log_w = None  # [3] float32 CUDA tensor; a tensor so torch.compile doesn't recompile per value
ACTIVE = False


def set_weights(weights: dict | None) -> None:
    global _log_w, ACTIVE
    if not weights:
        return
    for key in _weights:
        if key in weights and weights[key] is not None:
            _weights[key] = min(4.0, max(0.05, float(weights[key])))
    vals = torch.tensor([_weights["sink"], _weights["recent"], _weights["current"]]).log()
    if _log_w is None:
        _log_w = vals.cuda()
    else:
        _log_w.copy_(vals)
    ACTIVE = any(w != 1.0 for w in _weights.values())


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
