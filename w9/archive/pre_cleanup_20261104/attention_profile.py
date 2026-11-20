
import math
import torch
from torch.profiler import profile, ProfilerActivity, record_function

device = "cuda"
B, H, S, D = 1, 8, 1024, 64

q = torch.randn(B, H, S, D, device=device)
k = torch.randn(B, H, S, D, device=device)
v = torch.randn(B, H, S, D, device=device)

mask = torch.ones(S, S, device=device, dtype=torch.bool).triu(1)

def attention(q, k, v):
    with record_function("01_qk_matmul"):
        scores = q @ k.transpose(-2, -1)

    with record_function("02_scale_and_mask"):
        scores = scores / math.sqrt(D)
        scores = scores.masked_fill_(mask, float("-inf"))

    with record_function("03_softmax"):
        weights = torch.softmax(scores, dim=-1)

    with record_function("04_av_matmul"):
        output = weights @ v

    return output

# 预热，避免首次初始化影响观察
for _ in range(10):
    attention(q, k, v)

torch.cuda.synchronize()

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ]
) as prof:
    attention(q, k, v)
    torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_device_time_total",
    row_limit=25
))

prof.export_chrome_trace("w9_attention_inplace_trace.json")

