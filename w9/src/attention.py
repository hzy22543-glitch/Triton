"""W9 causal attention, FP32 forward only; no profiler in benchmark path."""
from contextlib import nullcontext
import math
import torch


def make_inputs(sequence, seed=42, batch=1, heads=8, dim=64):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, heads, sequence, dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.float32,
                          generator=generator) for _ in range(3)]
    mask = torch.ones(sequence, sequence, device="cuda", dtype=torch.bool).triu(1)
    return q, k, v, mask


@torch.no_grad()
def attention(q, k, v, mask, inplace=False, annotate=False):
    def region(name):
        return torch.profiler.record_function(name) if annotate else nullcontext()
    with region("01_qk_matmul"):
        scores = q @ k.transpose(-2, -1)
    with region("02_scale_and_mask"):
        scores = scores / math.sqrt(q.shape[-1])
        if inplace:
            scores.masked_fill_(mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask, float("-inf"))
    with region("03_softmax"):
        weights = torch.softmax(scores, dim=-1)
    with region("04_av_matmul"):
        return weights @ v

