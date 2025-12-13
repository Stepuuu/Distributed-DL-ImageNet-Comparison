#!/bin/bash
# ImageNet数据集准备脚本
# 将Kaggle下载的数据集解压并构造成PyTorch ImageFolder格式

set -e

echo "=========================================="
echo "ImageNet 数据集准备"
echo "=========================================="
echo ""

# 检查zip文件
if [ ! -f "imagenet-object-localization-challenge.zip" ]; then
    echo "错误: 找不到 imagenet-object-localization-challenge.zip"
    exit 1
fi

echo "[1/5] 解压数据集（这可能需要20-30分钟）..."
unzip -q imagenet-object-localization-challenge.zip -d imagenet_raw
echo "✓ 解压完成"
echo ""

# 检查解压后的结构
echo "[2/5] 检查解压后的目录结构..."
ls -lh imagenet_raw/
echo ""

# 创建训练目录结构
echo "[3/5] 创建训练目录结构..."
mkdir -p train val

# 处理训练集
echo "[4/5] 处理训练集..."
if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/train" ]; then
    # 检查是tar文件还是已经解压的目录
    TAR_COUNT=$(ls imagenet_raw/ILSVRC/Data/CLS-LOC/train/*.tar 2>/dev/null | wc -l)
    DIR_COUNT=$(ls -d imagenet_raw/ILSVRC/Data/CLS-LOC/train/*/ 2>/dev/null | wc -l)
    
    if [ $TAR_COUNT -gt 0 ]; then
        # 情况1: 有tar文件，需要解压
        echo "找到 ${TAR_COUNT} 个类别的tar文件，开始解压..."
        cd imagenet_raw/ILSVRC/Data/CLS-LOC/train
        for tar_file in *.tar; do
            class_name="${tar_file%.tar}"
            echo "  解压类别: ${class_name}"
            mkdir -p "../../../../../train/${class_name}"
            tar -xf "${tar_file}" -C "../../../../../train/${class_name}"
        done
        cd ../../../../..
        echo "✓ 训练集解压完成"
    elif [ $DIR_COUNT -gt 0 ]; then
        # 情况2: 已经是目录结构（Kaggle版本），直接创建软链接
        echo "检测到已解压的目录结构 (${DIR_COUNT} 个类别)"
        echo "创建软链接到 train/ ..."
        ln -sf "$PWD/imagenet_raw/ILSVRC/Data/CLS-LOC/train" train
        echo "✓ 训练集已链接 (软链接模式，节省磁盘空间)"
    else
        echo "警告: train目录既没有tar文件也没有子目录"
    fi
else
    echo "警告: 未找到训练集目录"
fi
echo ""

# 处理验证集
echo "[5/5] 处理验证集..."
if [ -d "imagenet_raw/ILSVRC/Data/CLS-LOC/val" ]; then
    # 检查验证集是否已经按类别组织
    VAL_SUBDIRS=$(ls -d imagenet_raw/ILSVRC/Data/CLS-LOC/val/*/ 2>/dev/null | wc -l)
    
    if [ $VAL_SUBDIRS -gt 10 ]; then
        # 已经是目录结构，直接软链接
        echo "检测到验证集已按类别组织 (${VAL_SUBDIRS} 个类别)"
        ln -sf "$PWD/imagenet_raw/ILSVRC/Data/CLS-LOC/val" val
        echo "✓ 验证集已链接 (软链接模式)"
    else
        # 验证集是扁平结构，需要使用valprep.sh组织
        echo "验证集为扁平结构，需要按类别组织..."
        mkdir -p val
        cp -r imagenet_raw/ILSVRC/Data/CLS-LOC/val/* val/
        
        if [ -f "valprep.sh" ]; then
            echo "使用 valprep.sh 组织验证集..."
            cd val
            bash ../valprep.sh
            cd ..
            echo "✓ 验证集已组织完成"
        else
            echo "警告: 未找到 valprep.sh，验证集未按类别组织"
        fi
    fi
else
    echo "警告: 未找到验证集目录"
fi
echo ""

# 显示最终结构
echo "=========================================="
echo "数据集准备完成！"
echo "=========================================="
echo ""
echo "目录结构:"
echo "  train/ - $(find train -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l) 个类别"
echo "  val/   - $(find val -maxdepth 1 -type d 2>/dev/null | tail -n +2 | wc -l) 个类别"
echo ""
echo "训练集样本数: $(find train -type f 2>/dev/null | wc -l)"
echo "验证集样本数: $(find val -type f 2>/dev/null | wc -l)"
echo ""
echo "现在可以运行训练脚本了！"
echo "示例: bash run_all_experiments.sh"
