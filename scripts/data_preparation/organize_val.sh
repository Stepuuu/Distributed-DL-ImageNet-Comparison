#!/bin/bash
# 组织验证集：将扁平的50000张图片按类别分类到1000个文件夹

set -e

echo "=========================================="
echo "组织 ImageNet 验证集"
echo "=========================================="
echo ""

VAL_DIR="imagenet_raw/ILSVRC/Data/CLS-LOC/val"

# 检查验证集目录
if [ ! -d "$VAL_DIR" ]; then
    echo "错误: 验证集目录不存在: $VAL_DIR"
    exit 1
fi

# 检查是否已经组织好
SUBDIRS=$(find "$VAL_DIR" -maxdepth 1 -type d | tail -n +2 | wc -l)
if [ $SUBDIRS -gt 100 ]; then
    echo "✓ 验证集已按类别组织（${SUBDIRS}个子目录）"
    echo "删除旧的 val/ 软链接并重新创建..."
    [ -L "val" ] && rm val
    ln -sf "$PWD/$VAL_DIR" val
    echo "✓ 完成"
    exit 0
fi

echo "验证集当前为扁平结构，需要组织..."
echo "图片总数: $(find "$VAL_DIR" -maxdepth 1 -name "*.JPEG" 2>/dev/null | wc -l)"
echo ""

# 检查是否有 valprep.sh
if [ ! -f "valprep.sh" ]; then
    echo "错误: 未找到 valprep.sh 脚本"
    echo "请确保 valprep.sh 在当前目录"
    exit 1
fi

# 创建临时目录用于组织验证集
echo "[1/3] 创建临时验证集目录..."
VAL_ORGANIZED="val_organized"
rm -rf "$VAL_ORGANIZED"
mkdir -p "$VAL_ORGANIZED"

# 复制所有图片到临时目录
echo "[2/3] 复制图片到临时目录（这需要几分钟）..."
cp -r "$VAL_DIR"/*.JPEG "$VAL_ORGANIZED/"

# 在临时目录中运行 valprep.sh
echo "[3/3] 使用 valprep.sh 组织验证集..."
cd "$VAL_ORGANIZED"
bash ../valprep.sh
cd ..

echo ""
echo "✓ 验证集已组织完成"
echo ""

# 检查组织后的结构
CLASSES=$(find "$VAL_ORGANIZED" -maxdepth 1 -type d | tail -n +2 | wc -l)
echo "类别数: ${CLASSES}"

# 删除旧的 val 软链接
[ -L "val" ] && rm val
[ -d "val" ] && [ ! -L "val" ] && echo "警告: val/ 是目录不是软链接，请手动处理" && exit 1

# 创建新的软链接指向组织好的验证集
ln -sf "$PWD/$VAL_ORGANIZED" val

echo ""
echo "=========================================="
echo "✓ 验证集准备完成！"
echo "=========================================="
echo ""
echo "目录结构:"
ls -lh val | head -15

echo ""
echo "可以开始训练了！"
