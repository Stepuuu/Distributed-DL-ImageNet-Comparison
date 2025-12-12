# 分布式训练性能对比实验指南

## 📋 实验概述

本实验对比三种分布式训练方法的性能：
1. **Baseline DDP** - PyTorch原生分布式数据并行（使用All-Reduce）
2. **Manual All-Reduce** - 手动实现的All-Reduce梯度同步
3. **Parameter Server (PS)** - 参数服务器架构（Rank 0为Server，其余为Workers）

---

## 🔧 环境准备

### 1. 检查GPU可用性
```bash
nvidia-smi
```
确保至少有 **4个可用GPU**。

### 2. 检查必要文件
确保以下文件存在：
- `baseline_multi_card.py` - Baseline DDP训练脚本
- `all_reduce_train.py` - Manual All-Reduce训练脚本
- `ps_train.py` - Parameter Server训练脚本
- `analyze_results.py` - 性能分析和可视化脚本
- `resnet50-0676ba61.pth` - ResNet50预训练权重

### 3. 数据集准备
确保ImageNet数据集路径正确：
```bash
# 检查数据集结构
ls -la ./train/
ls -la ./val/
```

如果数据集不在当前目录，需要在运行命令中指定 `--data-dir` 参数。

### 4. 安装依赖包（如果需要）
```bash
pip install matplotlib numpy tqdm
```

---

## 🚀 运行实验

### 实验 A: Baseline DDP (PyTorch原生DDP)

**命令:**
```bash
torchrun --nproc_per_node=4 baseline_multi_card.py
```

**如果需要自定义数据路径:**
```bash
torchrun --nproc_per_node=4 baseline_multi_card.py
```
（注意：此脚本的数据路径在代码内部通过 `train()` 函数的 `path` 参数设置，默认为 `./`）

**预期输出:**
- 每个epoch的训练进度条（带loss、accuracy、throughput）
- 每个epoch的训练和验证指标摘要
- 最终生成 `results_baseline_ddp.json`

**运行时间:** 约 10-30 分钟（取决于数据集大小和硬件）

---

### 实验 B: Manual All-Reduce

**命令:**
```bash
torchrun --nproc_per_node=4 all_reduce_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl
```

**参数说明:**
- `--epochs 3`: 训练3个epoch
- `--batch-size 64`: 每个进程的batch size
- `--workers 16`: DataLoader的worker数量
- `--data-dir ./`: 数据集根目录
- `--backend nccl`: 使用NCCL通信后端（GPU必需）

**预期输出:**
- 每个epoch的训练进度条（只有Rank 0显示）
- 每个epoch的训练指标摘要（Loss、Accuracy、Throughput、通信时间）
- 最终生成 `results_all_reduce.json`

**运行时间:** 约 10-30 分钟

---

### 实验 C: Parameter Server

**命令:**
```bash
torchrun --nproc_per_node=4 ps_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl
```

**参数说明:**
- `--epochs 3`: 训练3个epoch
- `--batch-size 64`: 每个Worker的batch size
- `--workers 16`: DataLoader的worker数量
- `--data-dir ./`: 数据集根目录
- `--backend nccl`: 使用NCCL通信后端

**架构说明:**
- **Rank 0**: 参数服务器（PS），负责参数存储和更新
- **Rank 1-3**: Workers，负责训练和梯度计算
- 通信流程：PS广播参数 → Workers计算梯度 → Workers发送梯度 → PS聚合并更新

**预期输出:**
- Worker端（Rank 1）显示训练进度条和详细指标
- PS端（Rank 0）显示批次处理进度和通信时间统计
- Workers显示训练和验证准确率
- 最终生成 `results_ps.json`

**运行时间:** 约 10-30 分钟（可能比DDP慢，因为通信模式不同）

---

## 📊 性能分析与可视化

### 运行分析脚本

所有三个实验完成后，运行分析脚本生成对比结果：

```bash
python analyze_results.py
```

**输出内容:**
1. **终端输出:**
   - 性能对比表格（吞吐量、训练时间、准确率）
   - 加速比分析（相对于Baseline DDP）

2. **生成的图表** (保存在 `./plots/` 目录):
   - `throughput_comparison.png` - 吞吐量对比柱状图
   - `training_time_comparison.png` - 训练时间对比柱状图
   - `loss_curves.png` - Loss收敛曲线对比
   - `accuracy_curves.png` - 准确率曲线对比
   - `communication_overhead.png` - 通信开销对比（PS vs All-Reduce）

3. **文本报告:**
   - `performance_report.txt` - 详细的性能报告

---

## 📈 结果文件说明

### JSON结果文件结构

每个实验会生成一个JSON文件，包含：

**`results_baseline_ddp.json`:**
```json
{
  "method": "Baseline DDP (PyTorch)",
  "world_size": 4,
  "batch_size": 64,
  "start_time": "2025-12-12 10:00:00",
  "end_time": "2025-12-12 10:15:00",
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 0.5234,
      "train_accuracy": 85.23,
      "train_time": 300.5,
      "train_throughput": 512.3,
      "avg_batch_time": 0.125,
      "val_loss": 0.4567,
      "val_accuracy": 87.45
    }
  ],
  "summary": {
    "avg_train_throughput": 510.2,
    "avg_train_time_per_epoch": 305.3,
    "best_val_accuracy": 88.67
  },
  "all_losses": [...]
}
```

**`results_all_reduce.json`:**
```json
{
  "method": "Manual All-Reduce",
  "world_size": 4,
  "batch_size": 64,
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 0.5123,
      "train_accuracy": 85.67,
      "train_time": 310.2,
      "train_throughput": 498.3,
      "avg_batch_time": 0.128,
      "avg_comm_time": 0.015,
      "all_losses": [...]
    }
  ],
  "summary": {
    "avg_train_throughput": 495.6,
    "avg_train_time_per_epoch": 312.1,
    "final_train_accuracy": 86.23
  }
}
```

**`results_ps.json`:**
```json
{
  "method": "Parameter Server (PS)",
  "world_size": 4,
  "num_workers": 3,
  "batch_size": 64,
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 0.5345,
      "train_accuracy": 84.56,
      "train_time": 350.8,
      "train_throughput": 440.2,
      "avg_batch_time": 0.145,
      "avg_comm_time_pull": 0.025,
      "avg_comm_time_push": 0.018,
      "val_loss": 0.4789,
      "val_accuracy": 86.34
    }
  ],
  "summary": {
    "avg_train_throughput": 438.5,
    "avg_train_time_per_epoch": 352.3,
    "avg_comm_time_pull": 0.024,
    "avg_comm_time_push": 0.019,
    "best_val_accuracy": 87.12
  },
  "all_losses": [...]
}
```

---

## 📝 撰写报告指南

### Section 3.2 (Implementation) - 实现细节

#### Parameter Server 架构分析

**1. 拓扑结构:**
```
角色分配:
- Rank 0: Parameter Server (PS)
  - 存储全局模型参数
  - 接收所有Workers的梯度
  - 聚合梯度并更新参数
  
- Rank 1-N: Workers
  - 加载数据并进行前向传播
  - 计算梯度
  - 与PS进行参数和梯度通信
```

**关键代码位置 (ps_train.py):**
- **角色判定**: 第20行 `is_ps = (rank == 0)`
- **参数拉取**: 第120-122行 (Workers从PS接收参数)
  ```python
  for param in model.parameters():
      dist.recv(param.data, src=0)
  ```
- **梯度推送**: 第136-138行 (Workers发送梯度给PS)
  ```python
  for param in model.parameters():
      dist.send(param.grad.data, dst=0)
  ```
- **参数广播**: 第178-181行 (PS发送参数给所有Workers)
  ```python
  for param in model.parameters():
      for worker_rank in range(1, world_size):
          dist.send(param.data, dst=worker_rank)
  ```
- **梯度聚合**: 第184-193行 (PS接收并平均梯度)
  ```python
  for param in model.parameters():
      grad_data = torch.zeros_like(param.data)
      for worker_rank in range(1, world_size):
          worker_grad = torch.zeros_like(param.data)
          dist.recv(worker_grad, src=worker_rank)
          grad_data += worker_grad
      grad_data /= num_workers
      param.grad = grad_data
  ```

**2. 通信流程 (同步PS):**
```
每个训练批次:
1. PS → Workers: 广播最新参数
2. Workers: 本地前向传播和反向传播
3. Workers → PS: 发送梯度
4. PS: 聚合梯度 (求平均)
5. PS: 更新参数
6. 重复步骤1
```

**3. 优化点分析:**

原始代码的问题：
- 使用逐参数逐Worker的点对点通信（`dist.send/recv`），效率较低
- 没有使用PyTorch的集合通信原语（`broadcast`, `reduce`）
- 每个参数都单独通信，无法利用批量传输优势

改进建议：
- 使用 `dist.broadcast()` 替代循环发送参数
- 使用 `dist.reduce()` 聚合梯度
- 考虑参数打包传输减少通信次数

---

### Section 4.2 (Evaluation) - 性能评估

#### 对比表格模板

从 `performance_report.txt` 或终端输出提取数据，填入以下表格：

| 方法 | 平均吞吐量 (img/s) | 平均训练时间/epoch (s) | 最佳准确率 (%) | 相对加速比 |
|------|-------------------|----------------------|--------------|----------|
| Baseline DDP | XXX.XX | XXX.XX | XX.XX | 1.00x (基准) |
| Manual All-Reduce | XXX.XX | XXX.XX | XX.XX | X.XXx |
| Parameter Server | XXX.XX | XXX.XX | XX.XX | X.XXx |

#### 通信开销对比

从结果JSON中提取：

| 方法 | 参数拉取 (ms) | 梯度推送 (ms) | All-Reduce (ms) | 总通信时间 (ms) |
|------|-------------|--------------|----------------|----------------|
| Baseline DDP | - | - | - | (隐式，由PyTorch处理) |
| Manual All-Reduce | - | - | XX.XX | XX.XX |
| Parameter Server | XX.XX | XX.XX | - | XX.XX + XX.XX |

#### 分析要点

1. **吞吐量分析:**
   - DDP通常最快（使用高度优化的NCCL All-Reduce）
   - PS可能较慢（点对点通信开销大）
   - 解释差异原因

2. **收敛性分析:**
   - 对比Loss曲线图
   - 三种方法的收敛速度是否一致？
   - 最终准确率差异

3. **通信效率:**
   - PS的通信是同步且串行的
   - All-Reduce可以利用树形拓扑和带宽聚合
   - 从图表 `communication_overhead.png` 中分析

4. **可扩展性:**
   - PS架构在大规模场景下的瓶颈（单点PS）
   - DDP的扩展性更好

---

## 🔍 故障排查

### 常见问题

**1. CUDA Out of Memory**
```bash
# 解决方案：减小batch size
torchrun --nproc_per_node=4 ps_train.py --batch-size 32
```

**2. 数据集路径错误**
```bash
# 检查数据集
ls ./train/ ./val/

# 或指定正确路径
torchrun --nproc_per_node=4 ps_train.py --data-dir /path/to/imagenet
```

**3. NCCL通信超时**
```bash
# 增加超时时间
export NCCL_TIMEOUT=1800
torchrun --nproc_per_node=4 ps_train.py
```

**4. 可视化脚本报错（缺少matplotlib）**
```bash
pip install matplotlib numpy
```

**5. 权重文件不存在**
```python
# 修改脚本使用在线下载（如果网络允许）
# 将 model = models.resnet50(weights=None) 和 state_dict加载部分
# 改为：
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
```

---

## 📤 提交材料清单

实验完成后，准备以下文件用于报告：

### 必需文件:
1. ✅ `results_baseline_ddp.json`
2. ✅ `results_all_reduce.json`
3. ✅ `results_ps.json`
4. ✅ `performance_report.txt`
5. ✅ `plots/` 目录下的所有图表：
   - `throughput_comparison.png`
   - `training_time_comparison.png`
   - `loss_curves.png`
   - `accuracy_curves.png`
   - `communication_overhead.png`

### 源代码（已优化）:
6. ✅ `baseline_multi_card.py`
7. ✅ `all_reduce_train.py`
8. ✅ `ps_train.py`
9. ✅ `analyze_results.py`

### 报告内容建议:

**Section 3.2 (Implementation):**
- Parameter Server架构图
- 关键代码片段（角色分配、通信流程）
- 与DDP的架构对比

**Section 4.2 (Evaluation):**
- 性能对比表格
- 吞吐量和训练时间对比图
- Loss收敛曲线图
- 通信开销分析
- 结论与分析

---

## 🎯 快速实验流程（完整命令）

如果一切环境就绪，可以按顺序执行：

```bash
# 1. 清理旧结果
rm -f results_*.json performance_report.txt
rm -rf plots/

# 2. 运行三个实验（依次执行，每个约10-30分钟）
echo "Running Baseline DDP..."
torchrun --nproc_per_node=4 baseline_multi_card.py

echo "Running Manual All-Reduce..."
torchrun --nproc_per_node=4 all_reduce_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl

echo "Running Parameter Server..."
torchrun --nproc_per_node=4 ps_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl

# 3. 分析结果
echo "Analyzing results..."
python analyze_results.py

# 4. 查看生成的文件
echo "Generated files:"
ls -lh results_*.json performance_report.txt plots/*.png
```

---

## 📞 技术支持

如遇到问题，可以：
1. 检查本文档的"故障排查"章节
2. 查看脚本内的详细日志输出
3. 确认GPU和数据集配置正确

---

**祝实验顺利！🎉**
