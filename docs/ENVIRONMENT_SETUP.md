# 🔧 环境配置指南

## 快速开始

### 方法1：自动配置（推荐）

运行自动配置脚本：
```bash
cd /inspire/hdd/global_user/shengyang-253107100022/ai_gongcheng_hw_all/ai_gongcheng_hw3
./setup_environment.sh
```

这个脚本会自动：
1. 创建名为 `distributed_training` 的conda环境（Python 3.10）
2. 检测CUDA版本并安装对应的PyTorch
3. 安装所有需要的依赖包（tqdm, matplotlib, numpy）
4. 验证安装是否成功

**预计时间：5-10分钟**

---

### 方法2：手动配置

如果自动脚本有问题，可以手动执行以下步骤：

#### 步骤1：创建conda环境
```bash
conda create -n distributed_training python=3.10 -y
conda activate distributed_training
```

#### 步骤2：安装PyTorch

**如果你有CUDA 12.x：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**如果你有CUDA 11.x：**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**如果没有GPU或想用CPU：**
```bash
pip install torch torchvision torchaudio
```

#### 步骤3：安装其他依赖
```bash
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install tqdm matplotlib numpy
```

#### 步骤4：验证安装
```bash
python check_environment.py
```

---

## 验证安装

安装完成后，运行环境检查脚本：

```bash
conda activate distributed_training
python check_environment.py
```

你应该看到：
- ✓ CUDA可用
- ✓ 检测到 4 个GPU
- ✓ 所有依赖包已安装
- ✓ 必需文件存在

---

## 后续使用

### 每次使用前激活环境

```bash
conda activate distributed_training
```

### 运行实验

```bash
# 方法A：一键运行（推荐）
./run_all_experiments.sh

# 方法B：手动运行
torchrun --nproc_per_node=4 baseline_multi_card.py
torchrun --nproc_per_node=4 all_reduce_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl
torchrun --nproc_per_node=4 ps_train.py --epochs 3 --batch-size 64 --workers 16 --data-dir ./ --backend nccl
python analyze_results.py
```

---

## 依赖包清单

### 必需依赖
- **Python**: 3.10
- **PyTorch**: >=2.0.0 (带CUDA支持)
- **TorchVision**: >=0.15.0
- **tqdm**: >=4.65.0 (进度条)
- **matplotlib**: >=3.7.0 (可视化)
- **numpy**: >=1.24.0 (数值计算)

### 系统要求
- **GPU**: 建议4个NVIDIA GPU
- **CUDA**: 11.x 或 12.x
- **显存**: 每个GPU至少8GB（建议16GB）
- **内存**: 至少32GB
- **磁盘**: 至少50GB可用空间（用于数据集）

---

## 常见问题

### Q: conda命令找不到？
**A:** 确保已安装Anaconda或Miniconda，并且已经初始化：
```bash
# 初始化conda
conda init bash
# 重新加载shell配置
source ~/.bashrc
```

### Q: PyTorch安装失败？
**A:** 尝试使用清华镜像：
```bash
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: ImportError: No module named 'torch'？
**A:** 确保激活了正确的环境：
```bash
conda activate distributed_training
which python  # 确认使用的是环境中的python
```

### Q: CUDA不可用？
**A:** 检查：
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查PyTorch的CUDA版本
python -c "import torch; print(torch.version.cuda)"

# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: 环境已存在，如何重新创建？
**A:** 删除旧环境再创建：
```bash
conda deactivate
conda env remove -n distributed_training
./setup_environment.sh
```

---

## 环境管理

### 查看所有环境
```bash
conda env list
```

### 激活环境
```bash
conda activate distributed_training
```

### 退出环境
```bash
conda deactivate
```

### 删除环境
```bash
conda env remove -n distributed_training
```

### 导出环境配置
```bash
conda activate distributed_training
conda env export > environment.yml
```

### 从配置文件创建环境
```bash
conda env create -f environment.yml
```

---

## 快速命令速查

```bash
# 一次性配置（首次使用）
./setup_environment.sh

# 后续每次使用
conda activate distributed_training

# 检查环境
python check_environment.py

# 运行实验
./run_all_experiments.sh

# 退出环境
conda deactivate
```

---

## 故障排查

如果遇到问题：

1. **查看详细错误信息**
   - 脚本会显示详细的错误消息
   - 记录错误信息以便排查

2. **手动验证每一步**
   - 按照"方法2：手动配置"逐步执行
   - 确认每一步都成功

3. **检查系统要求**
   - 确认GPU驱动正常（nvidia-smi）
   - 确认有足够的磁盘空间
   - 确认网络连接正常（用于下载包）

4. **查看日志**
   ```bash
   # 查看pip安装日志
   pip install torch --verbose
   
   # 查看conda日志
   conda install pytorch --verbose
   ```

---

## 完成确认

环境配置成功后，你应该能看到：

```
==========================================
验证安装
==========================================
✓ PyTorch版本: 2.x.x
✓ TorchVision版本: 0.x.x
✓ CUDA可用: True
✓ CUDA版本: 12.x
✓ GPU数量: 4
  - GPU 0: NVIDIA A100 (或其他型号)
  - GPU 1: NVIDIA A100
  - GPU 2: NVIDIA A100
  - GPU 3: NVIDIA A100
✓ tqdm: 4.x.x
✓ matplotlib: 3.x.x
✓ numpy: 1.x.x

==========================================
✓ 环境配置完成！
==========================================
```

**如果看到上述信息，说明环境已配置成功，可以开始运行实验！**

---

## 下一步

环境配置完成后：

1. **阅读实验文档**
   ```bash
   cat README.md
   ```

2. **运行环境检查**
   ```bash
   python check_environment.py
   ```

3. **开始实验**
   ```bash
   ./run_all_experiments.sh
   ```

祝实验顺利！🚀
