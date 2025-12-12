# 🎓 分布式训练实验 - 开始指南

> **重要提示：请先阅读本文件！这是最重要的起始文档。**

---

## 📌 快速导航

你现在看到的是项目的入口文档。根据你的需求选择：

1. **我想快速开始实验** → 看下方"快速开始"部分
2. **我想了解所有改进内容** → 阅读 `工作说明_请先阅读.md`
3. **我想看详细的中文指南** → 阅读 `README_CN.md`
4. **我想看英文详细文档** → 阅读 `EXPERIMENT_GUIDE.md`
5. **我想看命令速查** → 查看 `QUICK_REFERENCE.txt`

---

## 🚀 快速开始（三步走）

### 步骤1：检查环境

确保你在正确的目录：
```bash
cd /inspire/hdd/global_user/shengyang-253107100022/ai_gongcheng_hw_all/ai_gongcheng_hw3
```

检查GPU（确保有4个可用GPU）：
```bash
nvidia-smi
```

检查数据集（确保存在）：
```bash
ls -la ./train/ ./val/
```

### 步骤2：运行实验

**方法A：一键运行（推荐）**
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```
这会自动运行所有三个实验并生成分析报告（预计30-90分钟）。

**方法B：手动运行（如果需要更多控制）**
```bash
# 实验A: Baseline DDP
torchrun --nproc_per_node=4 baseline_multi_card.py

# 实验B: Manual All-Reduce
torchrun --nproc_per_node=4 all_reduce_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl

# 实验C: Parameter Server (你负责的部分)
torchrun --nproc_per_node=4 ps_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl

# 生成分析报告
python analyze_results.py
```

### 步骤3：查看结果

```bash
# 查看性能对比报告
cat performance_report.txt

# 查看生成的图表
ls -lh plots/

# 查看JSON结果文件
ls -lh results_*.json
```

---

## 📊 你会得到什么

### 实验完成后的输出文件：

**结果数据：**
- `results_baseline_ddp.json` - Baseline DDP的完整指标
- `results_all_reduce.json` - Manual All-Reduce的完整指标
- `results_ps.json` - Parameter Server的完整指标

**分析报告：**
- `performance_report.txt` - 详细的性能对比报告（可直接用于报告）

**可视化图表（plots/目录）：**
- `throughput_comparison.png` - 吞吐量对比柱状图
- `training_time_comparison.png` - 训练时间对比柱状图
- `loss_curves.png` - Loss收敛曲线
- `accuracy_curves.png` - 准确率曲线
- `communication_overhead.png` - 通信开销对比

---

## 📝 撰写报告

### Section 3.2 (Implementation) - 你需要写的内容：

1. **Parameter Server架构说明**
   - 拓扑：Rank 0 = Server, Rank 1-3 = Workers
   - 同步训练模式

2. **关键代码位置**（从ps_train.py提取）
   - 角色分配：第20行
   - 参数拉取：第120-122行
   - 梯度推送：第136-138行
   - PS参数广播：第178-181行
   - PS梯度聚合：第184-193行

3. **通信流程**
   - PS广播参数 → Workers训练 → Workers发送梯度 → PS聚合更新

4. **与DDP对比**
   - DDP：All-Reduce，对等通信
   - PS：中心化，串行通信

📖 **详细说明见：`EXPERIMENT_GUIDE.md` 和 `工作说明_请先阅读.md`**

### Section 4.2 (Evaluation) - 你需要写的内容：

1. **性能对比表格**（从 performance_report.txt 提取）
2. **插入5张可视化图表**（plots/目录）
3. **性能分析**：
   - 吞吐量对比及原因
   - 收敛性分析
   - 通信开销分析
   - 可扩展性讨论
4. **结论和建议**

📖 **详细模板和要点见：`EXPERIMENT_GUIDE.md` 的 Section 4.2 部分**

---

## ⚙️ 常见问题

### Q: CUDA Out of Memory 怎么办？
A: 减小batch size
```bash
torchrun --nproc_per_node=4 ps_train.py --batch-size 32
```

### Q: 找不到数据集怎么办？
A: 检查路径或指定数据集位置
```bash
torchrun --nproc_per_node=4 ps_train.py --data-dir /path/to/imagenet
```

### Q: 我只有2个GPU怎么办？
A: 修改GPU数量参数
```bash
torchrun --nproc_per_node=2 ps_train.py
```

### Q: 想快速测试，不想等太久？
A: 只跑1个epoch
```bash
torchrun --nproc_per_node=4 ps_train.py --epochs 1
```

### Q: 缺少ResNet50权重文件？
A: 修改脚本使用在线下载（见 EXPERIMENT_GUIDE.md 故障排查部分）

---

## 📚 文档索引

| 文档名称 | 用途 | 推荐阅读顺序 |
|---------|------|------------|
| **README.md** (本文件) | 快速入门和导航 | ⭐ 第1个读 |
| **工作说明_请先阅读.md** | 详细的工作总结和改进说明 | ⭐ 第2个读 |
| **README_CN.md** | 中文快速操作指南 | ⭐ 第3个读 |
| **EXPERIMENT_GUIDE.md** | 英文详细实验指南 | 需要详细信息时读 |
| **QUICK_REFERENCE.txt** | 命令速查卡片 | 运行实验时参考 |
| **SUMMARY.md** | 技术总结和细节 | 深入了解时读 |

---

## 🎯 核心改进点

我为你做的主要工作：

### 1. 优化了三个训练脚本
- ✅ 添加详细的性能统计（吞吐量、训练时间、通信时间）
- ✅ 改进日志输出（清晰的进度条和摘要）
- ✅ 生成标准化的JSON结果文件
- ✅ 实时显示训练指标

### 2. 重构了Parameter Server实现（重点）
- ✅ 大幅提升代码质量和可读性
- ✅ 添加PS端和Worker端的性能统计
- ✅ 详细的通信时间分析
- ✅ 完善的训练和验证流程
- ✅ 清晰的角色标识和日志

### 3. 创建了自动化分析工具
- ✅ 一键生成性能对比报告
- ✅ 自动生成5张高质量图表
- ✅ 计算加速比和统计指标

### 4. 编写了完整的文档体系
- ✅ 中英文操作指南
- ✅ 详细的代码分析（供Section 3.2使用）
- ✅ 报告撰写模板（供Section 4.2使用）
- ✅ 故障排查手册

---

## ✅ 检查清单

在开始实验前：
- [ ] 确认在正确的目录
- [ ] 确认有4个可用GPU（`nvidia-smi`）
- [ ] 确认数据集存在（`ls ./train/ ./val/`）
- [ ] 确认PyTorch环境已激活

在实验后：
- [ ] 生成了3个JSON结果文件
- [ ] 生成了 performance_report.txt
- [ ] plots/目录下有5张图表
- [ ] 查看了报告内容

提交材料：
- [ ] 源代码（ps_train.py等）
- [ ] 实验结果（JSON文件和报告）
- [ ] 可视化图表（plots/目录）
- [ ] 完成Section 3.2和4.2的撰写

---

## 🆘 需要帮助？

1. **快速问题** → 查看 `QUICK_REFERENCE.txt`
2. **操作步骤** → 查看 `README_CN.md`
3. **详细指南** → 查看 `EXPERIMENT_GUIDE.md`
4. **工作总结** → 查看 `工作说明_请先阅读.md`
5. **技术细节** → 查看 `SUMMARY.md`

---

## 🎉 总结

一切都已准备就绪！你现在有：

- ✨ 三个优化好的训练脚本
- ✨ 自动化的分析工具
- ✨ 完整的文档体系
- ✨ 清晰的实验流程

**只需要运行实验，然后使用生成的结果撰写报告！**

---

## 🚀 立即开始

```bash
# 第一步：进入目录
cd /inspire/hdd/global_user/shengyang-253107100022/ai_gongcheng_hw_all/ai_gongcheng_hw3

# 第二步：检查GPU
nvidia-smi

# 第三步：运行实验
./run_all_experiments.sh

# 完成！查看结果
cat performance_report.txt
ls plots/
```

**祝实验顺利！加油！🎓**
