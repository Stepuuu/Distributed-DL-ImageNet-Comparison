#!/bin/bash
# ImageNet 解压恢复脚本 - 支持从中断处继续

set -e

echo "=========================================="
echo "检查解压状态并恢复..."
echo "=========================================="
echo ""

# 检查zip文件
if [ ! -f "imagenet-object-localization-challenge.zip" ]; then
    echo "错误: 找不到 imagenet-object-localization-challenge.zip"
    exit 1
fi

# 检查是否已有部分解压内容
if [ -d "imagenet_raw" ]; then
    CURRENT_SIZE=$(du -sh imagenet_raw | cut -f1)
    echo "发现已存在的 imagenet_raw/ 目录"
    echo "当前大小: ${CURRENT_SIZE}"
    echo ""
    
    # 检查ILSVRC目录是否完整
    if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/train" ]; then
        TRAIN_CLASSES=$(ls -d imagenet_raw/ILSVRC/Data/CLS-LOC/train/*/ 2>/dev/null | wc -l)
        echo "训练集已有 ${TRAIN_CLASSES} 个类别"
        
        if [ $TRAIN_CLASSES -ge 1000 ]; then
            echo "✓ 训练集看起来已完整（1000个类别）"
        else
            echo "! 训练集不完整，需要重新解压"
        fi
    else
        echo "! 训练集目录不存在或不完整"
    fi
    
    if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/val" ]; then
        VAL_IMAGES=$(ls imagenet_raw/ILSVRC/Data/CLS-LOC/val/*.JPEG 2>/dev/null | wc -l)
        VAL_DIRS=$(ls -d imagenet_raw/ILSVRC/Data/CLS-LOC/val/*/ 2>/dev/null | wc -l)
        echo "验证集已有 ${VAL_IMAGES} 个图片文件，${VAL_DIRS} 个子目录"
        
        if [ $VAL_IMAGES -ge 50000 ] || [ $VAL_DIRS -ge 1000 ]; then
            echo "✓ 验证集看起来已完整"
        fi
    fi
    
    echo ""
    read -p "是否删除现有目录并重新完整解压? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除 imagenet_raw/ ..."
        rm -rf imagenet_raw
        echo "✓ 已删除"
    else
        echo "保留现有目录"
        echo ""
        echo "如果解压不完整，建议删除后重新解压："
        echo "  rm -rf imagenet_raw"
        echo "  bash resume_unzip.sh"
        exit 0
    fi
fi

echo ""
echo "开始完整解压 (预计20-30分钟)..."
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 使用unzip解压
# -n: 不覆盖已存在的文件（支持续传，但unzip本身不完全支持断点）
# -q: 安静模式
# 对于真正的断点续传，最好是完整重新解压

unzip -o imagenet-object-localization-challenge.zip -d imagenet_raw

echo ""
echo "=========================================="
echo "解压完成！"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 验证结果
if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/train" ]; then
    TRAIN_CLASSES=$(ls -d imagenet_raw/ILSVRC/Data/CLS-LOC/train/*/ 2>/dev/null | wc -l)
    echo "✓ 训练集: ${TRAIN_CLASSES} 个类别"
fi

if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/val" ]; then
    VAL_COUNT=$(find imagenet_raw/ILSVRC/Data/CLS-LOC/val -type f -name "*.JPEG" 2>/dev/null | wc -l)
    echo "✓ 验证集: ${VAL_COUNT} 个图片"
fi

TOTAL_SIZE=$(du -sh imagenet_raw | cut -f1)
echo "✓ 总大小: ${TOTAL_SIZE}"
echo ""
echo "下一步: 创建软链接"
echo "  ln -sf \$PWD/imagenet_raw/ILSVRC/Data/CLS-LOC/train train"
echo "  ln -sf \$PWD/imagenet_raw/ILSVRC/Data/CLS-LOC/val val"
