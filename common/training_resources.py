"""记录当前训练进程的实测内存峰值，不调整资源保护或计算精度。"""

import os

import psutil
import torch

from artifacts import now, write_json


def record_resources(run, device):
    info = psutil.Process().memory_info()
    report = {"at": now(), "pid": os.getpid(), "working_set_bytes": info.rss,
              "peak_working_set_bytes": getattr(info, "peak_wset", info.rss),
              "private_bytes": getattr(info, "private", None),
              "peak_commit_bytes": getattr(info, "peak_pagefile", None),
              "available_physical_bytes": psutil.virtual_memory().available}
    if torch.device(device).type == "cuda":
        report.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved())
    write_json(run / "resource-usage.json", report)
    return report
