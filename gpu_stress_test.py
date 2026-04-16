#!/usr/bin/env python3
import argparse
import time
from typing import Iterable, List

import torch
import torch.multiprocessing as mp


def _parse_gpu_ids(text: str, gpu_count: int) -> List[int]:
    if text.lower() == "all":
        return list(range(gpu_count))
    ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    for gpu_id in ids:
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(f"非法 GPU ID: {gpu_id}，当前可用 GPU 数量: {gpu_count}")
    if not ids:
        raise ValueError("未提供有效 GPU ID")
    return ids


def _allocate_memory(device: torch.device, target_bytes: int, chunk_mb: int) -> List[torch.Tensor]:
    buffers: List[torch.Tensor] = []
    chunk_bytes = chunk_mb * 1024 * 1024
    chunk_elements = max(1, chunk_bytes // 2)  # float16: 2 bytes
    allocated = 0
    while allocated < target_bytes:
        remaining = target_bytes - allocated
        elements = min(chunk_elements, max(1, remaining // 2))
        try:
            buf = torch.empty(elements, dtype=torch.float16, device=device)
        except RuntimeError:
            break
        buffers.append(buf)
        allocated += elements * 2
    return buffers


def _run_compute(device_id: int, duration: int, matrix_size: int, mem_ratio: float, chunk_mb: int) -> None:
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    props = torch.cuda.get_device_properties(device_id)
    total_bytes = props.total_memory
    target_bytes = int(total_bytes * mem_ratio)

    buffers = _allocate_memory(device, target_bytes, chunk_mb)
    allocated = sum(buf.numel() * buf.element_size() for buf in buffers)

    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float16)

    start = time.time()
    steps = 0
    while time.time() - start < duration:
        c = torch.matmul(a, b)
        a, b = b, c
        steps += 1
    torch.cuda.synchronize(device)
    print(
        f"[GPU {device_id}] 显存占用约 {allocated / 1024 ** 3:.2f} GB / {total_bytes / 1024 ** 3:.2f} GB, "
        f"计算迭代次数: {steps}"
    )


def _run_single(device_id: int, duration: int, matrix_size: int, mem_ratio: float, chunk_mb: int) -> None:
    _run_compute(device_id, duration, matrix_size, mem_ratio, chunk_mb)


def _run_multi(device_ids: Iterable[int], duration: int, matrix_size: int, mem_ratio: float, chunk_mb: int) -> None:
    ctx = mp.get_context("spawn")
    processes = []
    for device_id in device_ids:
        p = ctx.Process(target=_run_compute, args=(device_id, duration, matrix_size, mem_ratio, chunk_mb))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch GPU 压力测试（显存 + 算力）")
    parser.add_argument("--mode", choices=["single", "multi"], default="single", help="single=单卡，multi=多卡")
    parser.add_argument("--gpu-id", type=int, default=0, help="单卡模式下使用的 GPU ID")
    parser.add_argument("--gpus", default="all", help='多卡模式下使用的 GPU 列表，如 "0,1,2" 或 "all"')
    parser.add_argument("--duration", type=int, default=60, help="测试时长（秒）")
    parser.add_argument("--matrix-size", type=int, default=8192, help="矩阵乘大小，越大算力压力越高")
    parser.add_argument("--mem-ratio", type=float, default=0.9, help="目标显存占比（0~1）")
    parser.add_argument("--chunk-mb", type=int, default=256, help="显存分块大小（MB）")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU，请确认 PyTorch 安装了 CUDA 版本并且 GPU 可用")
    if not (0 < args.mem_ratio <= 1):
        raise ValueError("--mem-ratio 必须在 (0, 1] 范围内")
    if args.matrix_size <= 0:
        raise ValueError("--matrix-size 必须为正整数")
    if args.chunk_mb <= 0:
        raise ValueError("--chunk-mb 必须为正整数")

    gpu_count = torch.cuda.device_count()
    if args.mode == "single":
        if args.gpu_id < 0 or args.gpu_id >= gpu_count:
            raise ValueError(f"非法 --gpu-id: {args.gpu_id}，当前可用 GPU 数量: {gpu_count}")
        _run_single(args.gpu_id, args.duration, args.matrix_size, args.mem_ratio, args.chunk_mb)
    else:
        device_ids = _parse_gpu_ids(args.gpus, gpu_count)
        _run_multi(device_ids, args.duration, args.matrix_size, args.mem_ratio, args.chunk_mb)


if __name__ == "__main__":
    main()
