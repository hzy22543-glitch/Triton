"""Record software/device metadata, not secrets or full environment variables."""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import platform
import subprocess
import sys
import torch


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


print(json.dumps({
    "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "platform": platform.platform(), "python": sys.version,
    "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    "nvidia_smi": command("nvidia-smi"), "nvcc": command("nvcc", "--version"),
    "power_mode": "not recorded; shared display GPU; no exclusive-GPU guarantee",
    "benchmark_tf32": False, "benchmark_dtype": "float32"
}, indent=2))

