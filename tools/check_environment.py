#!/usr/bin/env python3
"""
环境检查脚本 - 在运行实验前检查必要的配置
"""

import os
import sys

import torch


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_gpu():
    """检查GPU可用性"""
    print_section("GPU检查")

    if not torch.cuda.is_available():
        print("❌ CUDA不可用！请确保安装了支持GPU的PyTorch")
        return False

    num_gpus = torch.cuda.device_count()
    print("✓ CUDA可用")
    print(f"✓ 检测到 {num_gpus} 个GPU")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  - GPU {i}: {gpu_name}")

    if num_gpus < 4:
        print(f"⚠ 警告: 实验脚本配置为使用4个GPU，当前只有{num_gpus}个")
        print("  建议: 修改脚本中的 --nproc_per_node 参数")

    return True


def check_files():
    """检查必要文件"""
    print_section("文件检查")

    required_files = [
        "baseline_multi_card.py",
        "all_reduce_train.py",
        "ps_train.py",
        "analyze_results.py",
        "run_all_experiments.sh",
    ]

    optional_files = ["resnet50-0676ba61.pth"]

    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ 缺少必需文件: {file}")
            all_ok = False

    for file in optional_files:
        if os.path.exists(file):
            print(f"✓ {file} (可选)")
        else:
            print(f"⚠ 缺少可选文件: {file}")
            print("  说明: 如果没有此文件，脚本会尝试在线下载ResNet50权重")

    return all_ok


def check_data():
    """检查数据集"""
    print_section("数据集检查")

    data_paths = ["./train", "./val"]

    for path in data_paths:
        if os.path.exists(path) and os.path.isdir(path):
            num_classes = len(
                [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            )
            print(f"✓ {path}/ 存在 (包含 {num_classes} 个类别)")
        else:
            print(f"❌ {path}/ 不存在")
            print("  说明: 请确保数据集在当前目录，或使用 --data-dir 参数指定路径")
            return False

    return True


def check_dependencies():
    """检查Python依赖"""
    print_section("依赖包检查")

    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "tqdm": "tqdm",
        "matplotlib": "Matplotlib (用于可视化)",
        "numpy": "NumPy",
    }

    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ 缺少依赖: {name}")
            print(f"  安装命令: pip install {package}")
            all_ok = False

    return all_ok


def check_distributed():
    """检查分布式训练支持"""
    print_section("分布式支持检查")

    if torch.distributed.is_available():
        print("✓ PyTorch分布式支持已启用")

        # 检查NCCL后端
        if torch.distributed.is_nccl_available():
            print("✓ NCCL后端可用 (GPU通信)")
        else:
            print("⚠ NCCL后端不可用")

        # 检查Gloo后端
        if torch.distributed.is_gloo_available():
            print("✓ Gloo后端可用 (CPU通信)")
        else:
            print("⚠ Gloo后端不可用")
    else:
        print("❌ PyTorch分布式支持未启用")
        return False

    return True


def estimate_time():
    """估算实验时间"""
    print_section("时间估算")

    print("基于典型配置的预估时间（实际时间取决于硬件和数据集大小）:")
    print("  - Baseline DDP: 10-30分钟")
    print("  - Manual All-Reduce: 10-30分钟")
    print("  - Parameter Server: 10-30分钟")
    print("  - 总计: 30-90分钟")
    print("\n如果只是测试，可以将epochs改为1，大约5-15分钟每个实验")


def main():
    print("\n" + "=" * 80)
    print("  分布式训练实验环境检查")
    print("=" * 80)

    checks = [
        ("GPU", check_gpu),
        ("依赖包", check_dependencies),
        ("分布式支持", check_distributed),
        ("必需文件", check_files),
        ("数据集", check_data),
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()

    estimate_time()

    # 总结
    print_section("检查总结")

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{name:15} {status}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有检查通过！可以开始运行实验")
        print("\n快速启动命令:")
        print("  ./run_all_experiments.sh")
        print("\n或查看详细指南:")
        print("  cat README_CN.md")
    else:
        print("❌ 部分检查失败，请解决上述问题后再运行实验")
        print("\n常见解决方案:")
        print(
            "  1. 安装缺少的依赖: pip install torch torchvision tqdm matplotlib numpy"
        )
        print("  2. 检查数据集路径是否正确")
        print("  3. 确认GPU和CUDA环境配置")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
