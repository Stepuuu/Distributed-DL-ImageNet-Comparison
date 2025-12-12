#!/bin/bash
# 分布式训练实验自动化运行脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "分布式训练性能对比实验"
echo "=========================================="
echo ""

# 配置参数
NPROC=4           # GPU数量
EPOCHS=3          # 训练epoch数
BATCH_SIZE=64     # Batch size
WORKERS=16        # DataLoader workers
DATA_DIR="./"     # 数据集路径
BACKEND="nccl"    # 通信后端

# 创建结果目录
echo "[1/5] 准备结果目录..."
mkdir -p results/plots
rm -f results/*.json results/*.txt
echo "✓ 结果目录准备完成"
echo ""

# 实验A: Baseline DDP
echo "[2/5] 运行实验A: Baseline DDP (PyTorch原生)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC baseline_multi_card.py
echo ""
echo "✓ 实验A完成 - 结果保存到 results/results_baseline_ddp.json"
echo ""

# 实验B: Manual All-Reduce
echo "[3/5] 运行实验B: Manual All-Reduce"
echo "=========================================="
torchrun --nproc_per_node=$NPROC all_reduce_train.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --data-dir $DATA_DIR \
    --backend $BACKEND
echo ""
echo "✓ 实验B完成 - 结果保存到 results/results_all_reduce.json"
echo ""

# 实验C: Parameter Server
echo "[4/5] 运行实验C: Parameter Server"
echo "=========================================="
torchrun --nproc_per_node=$NPROC ps_train.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --data-dir $DATA_DIR \
    --backend $BACKEND
echo ""
echo "✓ 实验C完成 - 结果保存到 results/results_ps.json"
echo ""

# 分析结果
echo "=========================================="
echo "[5/5] 生成性能分析报告和可视化图表..."
echo "=========================================="
python analyze_results.py
echo ""

# 显示生成的文件
echo "=========================================="
echo "✓ 所有实验完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  结果文件 (results/):"
ls -lh results/*.json 2>/dev/null | awk '{print "    - " $9 " (" $5 ")"}'
echo ""
echo "  报告文件:"
echo "    - performance_report.txt"
echo ""
echo "  图表文件 (./plots/):"
ls -1 plots/*.png 2>/dev/null | awk '{print "    - " $1}'
echo ""
echo "=========================================="
echo "请查看 EXPERIMENT_GUIDE.md 了解如何使用这些结果撰写报告"
echo "=========================================="
