#!/bin/bash
# 环境配置和依赖安装脚本

echo "=========================================="
echo "分布式训练实验环境配置"
echo "=========================================="
echo ""

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda命令"
    echo "请先安装Anaconda或Miniconda"
    exit 1
fi

echo "[1/5] 检测当前环境..."
echo "当前conda环境: $CONDA_DEFAULT_ENV"
echo ""

# 环境名称
ENV_NAME="distributed_training"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[2/5] 检测到环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "使用现有环境"
    fi
else
    echo "[2/5] 准备创建新环境 '${ENV_NAME}'"
fi

# 创建新环境（如果不存在）
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "创建conda环境: ${ENV_NAME} (Python 3.10)"
    conda create -n ${ENV_NAME} python=3.10 -y
    if [ $? -ne 0 ]; then
        echo "❌ 创建环境失败"
        exit 1
    fi
    echo "✓ 环境创建成功"
fi
echo ""

# 激活环境
echo "[3/5] 激活环境 ${ENV_NAME}..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

if [ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]; then
    echo "❌ 环境激活失败"
    exit 1
fi
echo "✓ 环境已激活: ${ENV_NAME}"
echo ""

# 检查CUDA版本
echo "[4/5] 检查CUDA版本..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "检测到CUDA版本: ${CUDA_VERSION}"
    
    # 根据CUDA版本安装PyTorch
    if [[ "${CUDA_VERSION}" == "12."* ]]; then
        echo "安装PyTorch (CUDA 12.x)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "${CUDA_VERSION}" == "11."* ]]; then
        echo "安装PyTorch (CUDA 11.x)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠ 未识别的CUDA版本，安装CPU版本PyTorch"
        pip install torch torchvision torchaudio
    fi
else
    echo "⚠ 未检测到NVIDIA GPU，安装CPU版本PyTorch"
    pip install torch torchvision torchaudio
fi
echo ""

# 安装其他依赖
echo "[5/5] 安装其他依赖包..."
pip install tqdm matplotlib numpy -q
echo "✓ 依赖安装完成"
echo ""

# 验证安装
echo "=========================================="
echo "验证安装"
echo "=========================================="
python -c "
import torch
import torchvision
import tqdm
import matplotlib
import numpy

print(f'✓ PyTorch版本: {torch.__version__}')
print(f'✓ TorchVision版本: {torchvision.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA版本: {torch.version.cuda}')
    print(f'✓ GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'✓ tqdm: {tqdm.__version__}')
print(f'✓ matplotlib: {matplotlib.__version__}')
print(f'✓ numpy: {numpy.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 环境配置完成！"
    echo "=========================================="
    echo ""
    echo "使用方法："
    echo "  1. 激活环境: conda activate ${ENV_NAME}"
    echo "  2. 运行实验: ./run_all_experiments.sh"
    echo ""
    echo "当前环境已激活，可以直接运行实验"
else
    echo ""
    echo "❌ 安装验证失败，请检查错误信息"
    exit 1
fi
