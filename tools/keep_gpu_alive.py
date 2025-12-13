#!/usr/bin/env python3
"""
GPU保活脚本 - 防止因GPU利用率低而被自动停止
每30分钟运行5分钟的GPU计算任务以提升GPU利用率
按 Ctrl+C 停止
"""

import signal
import sys
import time
from datetime import datetime

import torch

# 全局标志用于优雅退出
running = True


def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    global running
    print("\n\n收到停止信号，正在安全退出...")
    running = False


def gpu_intensive_task(duration_minutes=5):
    """
    执行GPU密集型任务 - 使用所有可用GPU
    duration_minutes: 任务持续时间（分钟）
    """
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，脚本将退出")
        return False

    num_gpus = torch.cuda.device_count()
    duration_seconds = duration_minutes * 60

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 开始GPU计算任务，持续 {duration_minutes} 分钟..."
    )
    print(f"使用 {num_gpus} 个GPU进行并行计算")

    # 增大矩阵以提高GPU利用率和显存占用
    matrix_size = 12288  # 从8192增大到12288，约占用更多显存

    start_time = time.time()
    iteration = 0

    try:
        # 预先在每个GPU上分配大块显存（持续占用）
        print("正在分配GPU显存...")
        persistent_tensors = []
        for gpu_id in range(num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            # 每个GPU分配多个大矩阵（约占用8-10GB显存）
            tensors_per_gpu = []
            for _ in range(3):  # 3个大矩阵
                tensor = torch.randn(matrix_size, matrix_size, device=device)
                tensors_per_gpu.append(tensor)
            persistent_tensors.append(tensors_per_gpu)

        # 显示初始显存占用
        gpu_mems = [torch.cuda.memory_allocated(i) / 1024**3 for i in range(num_gpus)]
        mem_str = " | ".join([f"GPU{i}: {mem:.2f}GB" for i, mem in enumerate(gpu_mems)])
        print(f"显存已分配: {mem_str}")
        print()

        # 持续进行GPU密集计算
        while time.time() - start_time < duration_seconds and running:
            # 在所有GPU上并行执行计算
            for gpu_id in range(num_gpus):
                device = torch.device(f"cuda:{gpu_id}")

                # 使用预分配的张量进行计算
                a = persistent_tensors[gpu_id][0]
                b = persistent_tensors[gpu_id][1]

                # GPU密集型操作
                c = torch.matmul(a, b)
                d = torch.sin(c) + torch.cos(c)
                e = torch.tanh(d) * 0.5

                # 原地更新以保持显存占用
                persistent_tensors[gpu_id][2].copy_(e)

                del c, d, e

            iteration += 1

            # 每2秒显示一次进度
            if iteration % 2 == 0:
                elapsed = time.time() - start_time
                # 显示所有GPU的显存使用
                gpu_mems = [
                    torch.cuda.memory_allocated(i) / 1024**3 for i in range(num_gpus)
                ]
                mem_str = " | ".join(
                    [f"GPU{i}: {mem:.2f}GB" for i, mem in enumerate(gpu_mems)]
                )
                print(
                    f"  进度: {elapsed:.0f}s / {duration_seconds}s | "
                    f"迭代: {iteration} | {mem_str}"
                )

            # 检查是否需要退出
            if not running:
                break

        # 清理所有预分配的显存
        print("\n正在清理GPU显存...")
        for tensors_per_gpu in persistent_tensors:
            for tensor in tensors_per_gpu:
                del tensor
        del persistent_tensors

        # 清理GPU缓存
        torch.cuda.empty_cache()

        if running:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU任务完成！")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU任务被中断")

        return True

    except Exception as e:
        print(f"错误: {e}")
        torch.cuda.empty_cache()
        return False


def main():
    """主循环：每30分钟运行5分钟GPU任务"""
    global running

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print("GPU保活脚本已启动")
    print("=" * 70)
    print("策略: 每30分钟运行5分钟的GPU计算任务")
    print("按 Ctrl+C 停止脚本")
    print("=" * 70)
    print()

    # 检查GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到可用的GPU")
        sys.exit(1)

    print(f"检测到 {torch.cuda.device_count()} 个GPU:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    cycle = 0

    while running:
        cycle += 1
        print(f"\n{'=' * 70}")
        print(f"第 {cycle} 轮循环开始")
        print(f"{'=' * 70}")

        # 执行5分钟的GPU任务
        success = gpu_intensive_task(duration_minutes=5)

        if not running:
            break

        if not success:
            print("GPU任务执行失败，退出脚本")
            break

        # 等待25分钟（共30分钟周期）
        wait_minutes = 25
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] GPU任务完成，等待 {wait_minutes} 分钟..."
        )
        print("按 Ctrl+C 可随时停止")

        # 分段等待，以便能响应Ctrl+C
        for i in range(wait_minutes * 60):
            if not running:
                break
            time.sleep(1)

            # 每5分钟显示一次倒计时
            if (i + 1) % 300 == 0:
                remaining = wait_minutes - (i + 1) // 60
                print(f"  剩余等待时间: {remaining} 分钟")

        if not running:
            break

    print("\n" + "=" * 70)
    print("GPU保活脚本已安全退出")
    print("=" * 70)


if __name__ == "__main__":
    main()
