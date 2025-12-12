# 分布式训练实验 - 快速操作指南

## 🎯 三步完成所有实验

### 方法1：自动化运行（推荐）

```bash
# 给脚本添加执行权限
chmod +x run_all_experiments.sh

# 一键运行所有实验
./run_all_experiments.sh
```

这个脚本会自动：
1. 清理旧结果
2. 依次运行三个实验（Baseline DDP、All-Reduce、Parameter Server）
3. 生成性能分析报告和可视化图表

**预计总时间:** 30-90分钟（取决于数据集大小和硬件性能）

---

### 方法2：手动分步运行

如果需要更多控制，可以手动运行每个实验：

#### 第1步：运行Baseline DDP
```bash
torchrun --nproc_per_node=4 baseline_multi_card.py
```

#### 第2步：运行Manual All-Reduce
```bash
torchrun --nproc_per_node=4 all_reduce_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl
```

#### 第3步：运行Parameter Server
```bash
torchrun --nproc_per_node=4 ps_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl
```

#### 第4步：生成分析报告
```bash
python analyze_results.py
```

---

## 📊 查看结果

### 实验输出文件

实验完成后会生成：

**1. JSON结果文件（原始数据）:**
- `results_baseline_ddp.json` - Baseline DDP的详细指标
- `results_all_reduce.json` - Manual All-Reduce的详细指标
- `results_ps.json` - Parameter Server的详细指标

**2. 性能报告:**
- `performance_report.txt` - 文本格式的详细性能对比报告

**3. 可视化图表（./plots/ 目录）:**
- `throughput_comparison.png` - 吞吐量对比柱状图
- `training_time_comparison.png` - 训练时间对比柱状图
- `loss_curves.png` - Loss收敛曲线
- `accuracy_curves.png` - 准确率曲线
- `communication_overhead.png` - 通信开销对比

### 快速查看结果

```bash
# 查看性能对比表格（终端输出）
cat performance_report.txt | head -50

# 查看生成的图表
ls -lh plots/

# 在VS Code中预览图表
code plots/throughput_comparison.png
```

---

## 📝 撰写报告

### Section 3.2 (Implementation) - 需要的内容

**Parameter Server实现分析:**

1. **架构说明:**
   - Rank 0 = Parameter Server（参数服务器）
   - Rank 1-3 = Workers（工作进程）
   - 同步训练模式

2. **关键代码位置（ps_train.py）:**
   - 角色分配：第20行
   - 参数下载：第120-122行（Workers从PS拉取参数）
   - 梯度上传：第136-138行（Workers推送梯度给PS）
   - 参数广播：第178-181行（PS分发参数给Workers）
   - 梯度聚合：第184-193行（PS接收并平均梯度）

3. **通信流程图:**
```
每个训练批次:
PS ---[broadcast params]---> Workers (所有Worker)
Workers --[forward + backward]-->
Workers ---[send gradients]---> PS (逐个Worker)
PS --[aggregate & update]-->
重复
```

4. **与DDP的区别:**
   - DDP: 使用All-Reduce，所有进程对等
   - PS: 中心化架构，PS是瓶颈点
   - DDP通信更高效（树形拓扑、NCCL优化）

---

### Section 4.2 (Evaluation) - 需要的内容

**从 `performance_report.txt` 中提取关键数据:**

1. **性能对比表格:**

| 方法 | 平均吞吐量 (img/s) | 训练时间/epoch (s) | 最佳准确率 (%) |
|------|-------------------|-------------------|--------------|
| Baseline DDP | [从结果填入] | [从结果填入] | [从结果填入] |
| Manual All-Reduce | [从结果填入] | [从结果填入] | [从结果填入] |
| Parameter Server | [从结果填入] | [从结果填入] | [从结果填入] |

2. **插入可视化图表:**
   - 吞吐量对比图（`throughput_comparison.png`）
   - 训练时间对比图（`training_time_comparison.png`）
   - Loss收敛曲线（`loss_curves.png`）
   - 通信开销对比（`communication_overhead.png`）

3. **性能分析要点:**
   - **吞吐量:** DDP通常最高，PS最低（解释原因：通信模式）
   - **收敛性:** 观察Loss曲线，判断是否一致收敛
   - **通信开销:** PS的参数拉取+梯度推送 vs All-Reduce的同步时间
   - **可扩展性:** PS在大规模时的瓶颈

4. **结论:**
   - DDP更适合同构集群和高吞吐场景
   - PS适合异构环境或需要灵活参数管理的场景
   - 通信效率是关键性能差异来源

---

## 🔧 参数调整

如果需要调整实验参数：

### 修改GPU数量
```bash
# 使用2个GPU
torchrun --nproc_per_node=2 baseline_multi_card.py

# 使用8个GPU
torchrun --nproc_per_node=8 baseline_multi_card.py
```

### 修改Batch Size（如果显存不足）
```bash
torchrun --nproc_per_node=4 ps_train.py --batch-size 32
```

### 修改训练Epoch数（快速测试）
```bash
torchrun --nproc_per_node=4 ps_train.py --epochs 1
```

### 指定数据集路径
```bash
torchrun --nproc_per_node=4 ps_train.py --data-dir /path/to/imagenet
```

---

## ⚠️ 常见问题

### 1. CUDA Out of Memory
**解决方案:** 减小batch size
```bash
torchrun --nproc_per_node=4 ps_train.py --batch-size 32
```

### 2. 没有找到数据集
**检查数据集结构:**
```bash
ls ./train/ ./val/
```
**或指定正确路径:**
```bash
torchrun --nproc_per_node=4 ps_train.py --data-dir /correct/path/
```

### 3. 缺少ResNet50权重文件
如果 `resnet50-0676ba61.pth` 不存在，需要修改脚本使用在线下载：
- 将三个脚本中的权重加载部分改为：
```python
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# 删除或注释掉 state_dict加载相关代码
```

### 4. 分析脚本报错（缺少matplotlib）
```bash
pip install matplotlib numpy
```

---

## 📦 完整文件清单

实验前确保有以下文件：
- ✅ `baseline_multi_card.py` - DDP训练脚本
- ✅ `all_reduce_train.py` - All-Reduce训练脚本
- ✅ `ps_train.py` - Parameter Server训练脚本
- ✅ `analyze_results.py` - 性能分析脚本
- ✅ `run_all_experiments.sh` - 自动化运行脚本
- ✅ `EXPERIMENT_GUIDE.md` - 详细实验指南
- ✅ `README_CN.md` - 本文件（快速操作指南）

实验后生成的文件：
- 📊 `results_baseline_ddp.json`
- 📊 `results_all_reduce.json`
- 📊 `results_ps.json`
- 📄 `performance_report.txt`
- 📁 `plots/` 目录及其中的5张图表

---

## 🚀 完整命令速查

```bash
# 一键运行（推荐）
chmod +x run_all_experiments.sh && ./run_all_experiments.sh

# 或手动运行三个实验
torchrun --nproc_per_node=4 baseline_multi_card.py
torchrun --nproc_per_node=4 all_reduce_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl
torchrun --nproc_per_node=4 ps_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl

# 生成分析报告
python analyze_results.py

# 查看结果
cat performance_report.txt
ls -lh plots/
```

---

## 📞 需要帮助？

1. 查看详细指南：`EXPERIMENT_GUIDE.md`
2. 检查脚本日志输出（实时显示训练进度和指标）
3. 确认环境配置（GPU、数据集、依赖包）

**祝实验顺利！如有问题可随时查阅文档。** 🎉
