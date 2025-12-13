#!/bin/bash
# 使用 LOC_val_solution.csv 组织验证集
# 比 valprep.sh 更快更清晰

set -e

echo "=========================================="
echo "组织 ImageNet 验证集（使用标签文件）"
echo "=========================================="
echo ""

VAL_SOURCE="imagenet_raw/ILSVRC/Data/CLS-LOC/val"
VAL_LABELS="imagenet_raw/LOC_val_solution.csv"
VAL_TARGET="val_organized"

# 检查文件
if [ ! -d "$VAL_SOURCE" ]; then
    echo "错误: 验证集目录不存在: $VAL_SOURCE"
    exit 1
fi

if [ ! -f "$VAL_LABELS" ]; then
    echo "错误: 标签文件不存在: $VAL_LABELS"
    exit 1
fi

# 检查是否已经组织好（检查图片数量而不是文件夹数量）
if [ -d "$VAL_TARGET" ]; then
    EXISTING_IMAGES=$(find "$VAL_TARGET" -type f -name "*.JPEG" 2>/dev/null | wc -l)
    echo "检测到现有的 val_organized/ 目录"
    echo "  已有图片: ${EXISTING_IMAGES} / 50000"
    
    if [ $EXISTING_IMAGES -ge 49000 ]; then
        echo "✓ 验证集已组织完成（图片数量足够）"
        [ -L "val" ] && rm val
        ln -sf "$PWD/$VAL_TARGET" val
        echo "✓ val/ 软链接已更新"
        exit 0
    else
        echo "  图片不完整，将继续复制..."
    fi
fi

echo "[1/4] 创建目标目录..."
rm -rf "$VAL_TARGET"
mkdir -p "$VAL_TARGET"

echo "[2/4] 从标签文件提取类别列表..."
# 从CSV提取所有唯一的类别ID（第二列，空格分隔后的第一个）
tail -n +2 "$VAL_LABELS" | cut -d',' -f2 | awk '{print $1}' | sort -u > /tmp/val_classes.txt
NUM_CLASSES=$(wc -l < /tmp/val_classes.txt)
echo "找到 ${NUM_CLASSES} 个类别"

echo "[3/4] 创建类别文件夹..."
while read -r class_id; do
    mkdir -p "$VAL_TARGET/$class_id"
done < /tmp/val_classes.txt

echo "[4/4] 复制图片到对应类别文件夹..."
echo "这可能需要5-10分钟..."
echo "支持断点续传（跳过已存在的文件）..."

# 使用Python脚本处理（更快更可靠，支持断点续传）
python3 << 'PYTHON_SCRIPT'
import csv
import shutil
import os
from pathlib import Path

val_source = "imagenet_raw/ILSVRC/Data/CLS-LOC/val"
val_labels = "imagenet_raw/LOC_val_solution.csv"
val_target = "val_organized"

print("开始复制图片...")
copied = 0
skipped = 0
errors = 0

with open(val_labels, 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        image_id = row['ImageId']
        # 提取第一个类别ID（格式：n02666196 0 6 373 498）
        class_id = row['PredictionString'].split()[0]
        
        src = f"{val_source}/{image_id}.JPEG"
        dst_dir = f"{val_target}/{class_id}"
        dst = f"{dst_dir}/{image_id}.JPEG"
        
        # 断点续传：跳过已存在的文件
        if os.path.exists(dst):
            skipped += 1
            if skipped % 5000 == 0:
                print(f"  已跳过 {skipped} 张已存在的图片...")
            continue
        
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                copied += 1
                if copied % 1000 == 0:
                    print(f"  已复制 {copied} 张（已跳过 {skipped} 张）...")
            except Exception as e:
                print(f"错误: {src} -> {dst}: {e}")
                errors += 1
        else:
            print(f"警告: 找不到图片 {src}")
            errors += 1

print(f"\n完成！")
print(f"  本次复制: {copied} 张")
print(f"  已存在跳过: {skipped} 张")
print(f"  总计: {copied + skipped} / 50000 张")
if errors > 0:
    print(f"  错误: {errors} 个文件")
PYTHON_SCRIPT

echo ""
echo "✓ 验证集已组织完成"
echo ""

# 统计信息
CLASSES=$(find "$VAL_TARGET" -maxdepth 1 -type d | tail -n +2 | wc -l)
IMAGES=$(find "$VAL_TARGET" -type f -name "*.JPEG" | wc -l)
echo "统计信息:"
echo "  类别数: ${CLASSES}"
echo "  图片数: ${IMAGES}"

# 删除旧的 val 软链接并创建新的
[ -L "val" ] && rm val
[ -d "val" ] && [ ! -L "val" ] && echo "警告: val/ 是目录不是软链接，请手动处理" && exit 1

ln -sf "$PWD/$VAL_TARGET" val

echo ""
echo "=========================================="
echo "✓ 验证集准备完成！"
echo "=========================================="
echo ""
echo "val/ -> $VAL_TARGET/"
echo ""
echo "示例类别:"
ls "$VAL_TARGET" | head -10

echo ""
echo "可以开始训练了！"
