#!/bin/bash
# ImageNet 断点续传解压并完成数据准备
# 从中断处继续解压，然后创建训练所需的目录结构

set -e

echo "=========================================="
echo "ImageNet 数据集恢复与准备"
echo "=========================================="
echo ""

# 检查zip文件
if [ ! -f "imagenet-object-localization-challenge.zip" ]; then
    echo "错误: 找不到 imagenet-object-localization-challenge.zip"
    exit 1
fi

# 步骤1: 继续解压（跳过已存在的文件）
echo "[1/3] 从断点继续解压数据集..."
echo "使用 -n 参数跳过已解压的文件"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# -n: never overwrite existing files (跳过已存在)
# -d: 解压到指定目录
unzip -n imagenet-object-localization-challenge.zip -d imagenet_raw

echo ""
echo "✓ 解压完成"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 步骤2: 创建训练/验证集软链接
echo "[2/3] 创建训练和验证集目录..."

# 检查训练集
if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/train" ]; then
    TRAIN_COUNT=$(ls -d imagenet_raw/ILSVRC/Data/CLS-LOC/train/*/ 2>/dev/null | wc -l)
    echo "训练集: ${TRAIN_COUNT} 个类别"
    
    # 删除旧的软链接（如果存在）
    [ -L "train" ] && rm train
    [ -d "train" ] && echo "警告: train/ 已存在且不是软链接，请手动处理" || \
        ln -sf "$PWD/imagenet_raw/ILSVRC/Data/CLS-LOC/train" train
    
    echo "✓ 训练集已链接: train/ -> imagenet_raw/ILSVRC/Data/CLS-LOC/train"
else
    echo "✗ 错误: 训练集目录不存在"
    exit 1
fi

# 检查验证集
if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/val" ]; then
    VAL_COUNT=$(find imagenet_raw/ILSVRC/Data/CLS-LOC/val -maxdepth 1 -type f -name "*.JPEG" 2>/dev/null | wc -l)
    VAL_DIRS=$(ls -d imagenet_raw/ILSVRC/Data/CLS-LOC/val/*/ 2>/dev/null | wc -l)
    
    echo "验证集: ${VAL_COUNT} 个图片, ${VAL_DIRS} 个子目录"
    
    # 删除旧的软链接
    [ -L "val" ] && rm val
    
    if [ $VAL_DIRS -gt 10 ]; then
        # 验证集已按类别组织
        [ -d "val" ] && echo "警告: val/ 已存在且不是软链接，请手动处理" || \
            ln -sf "$PWD/imagenet_raw/ILSVRC/Data/CLS-LOC/val" val
        echo "✓ 验证集已链接: val/ -> imagenet_raw/ILSVRC/Data/CLS-LOC/val"
    else
        # 验证集是扁平结构，需要组织（通常Kaggle版本已组织好）
        echo "注意: 验证集为扁平结构，创建软链接"
        [ -d "val" ] && echo "警告: val/ 已存在且不是软链接，请手动处理" || \
            ln -sf "$PWD/imagenet_raw/ILSVRC/Data/CLS-LOC/val" val
        echo "✓ 验证集已链接: val/ -> imagenet_raw/ILSVRC/Data/CLS-LOC/val"
    fi
else
    echo "✗ 错误: 验证集目录不存在"
    exit 1
fi

echo ""

# 步骤3: 验证并显示最终状态
echo "[3/3] 验证数据集结构..."
echo ""

echo "目录结构:"
ls -lh train val 2>/dev/null

echo ""
echo "统计信息:"
TRAIN_CLASSES=$(find train -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l)
TRAIN_IMAGES=$(find train -type f -name "*.JPEG" 2>/dev/null | wc -l)
echo "  训练集: ${TRAIN_CLASSES} 个类别, ${TRAIN_IMAGES} 张图片"

VAL_CLASSES=$(find val -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l)
VAL_IMAGES=$(find val -type f -name "*.JPEG" 2>/dev/null | wc -l)
echo "  验证集: ${VAL_CLASSES} 个类别, ${VAL_IMAGES} 张图片"

TOTAL_SIZE=$(du -sh imagenet_raw 2>/dev/null | cut -f1)
echo "  总大小: ${TOTAL_SIZE}"

echo ""
echo "=========================================="
echo "✓ 数据集准备完成！"
echo "=========================================="
echo ""
echo "可以开始训练了:"
echo "  单个实验: torchrun --nproc_per_node=4 baseline_multi_card.py"
echo "  全部实验: bash run_all_experiments.sh"
echo ""
echo "提示: train/ 和 val/ 是软链接，不占用额外磁盘空间"
