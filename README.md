# PyTorch Distributed Training Benchmark

A comprehensive comparison of distributed training methods for deep learning on ImageNet ResNet50, featuring manual implementations of Parameter Server, All-Reduce synchronization, and comparison with PyTorch native DDP (DistributedDataParallel).

## 📁 Project Structure

```
PyTorch-Distributed-Training-Benchmark/
├── README.md                          # Main project documentation (English)
├── .gitignore                         # Git ignore rules
├── train/                             # Symlink to ImageNet training data
├── val/                               # Symlink to ImageNet validation data
│
├── scripts/
│   ├── training/                      # Training implementations
│   │   ├── baseline_multi_card.py     # Standard PyTorch DDP baseline
│   │   ├── all_reduce_train.py        # Manual All-Reduce implementation
│   │   ├── ps_train.py                # Parameter Server implementation
│   │   ├── single_card.py             # Single GPU baseline
│   │   └── all_reduce_*.py            # Advanced variants
│   │
│   ├── data_preparation/              # Dataset preparation scripts
│   │   ├── prepare_imagenet.sh        # Initial ImageNet extraction
│   │   ├── organize_val_fast.sh       # Validation set organization
│   │   ├── continue_setup.sh          # Resume interrupted extraction
│   │   └── *.sh                       # Other utility scripts
│   │
│   └── analysis/                      # Results analysis
│       ├── run_all_experiments.sh     # Automated experiment runner
│       └── analyze_results.py         # Performance analysis & plots
│
├── tools/                             # Development utilities
│   ├── setup_environment.sh           # Environment setup
│   ├── requirements.txt               # Python dependencies
│   ├── check_environment.py           # Environment validation
│   └── keep_gpu_alive.py              # GPU utilization maintenance
│
└── docs/                              # Documentation
    ├── EXPERIMENT_GUIDE.md            # Detailed experiment guide
    ├── ENVIRONMENT_SETUP.md           # Setup instructions
    ├── FILES.md                       # File descriptions
    ├── README_CN.md                   # Chinese documentation
    └── *.md                           # Other docs
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: 4x NVIDIA GPUs (tested on RTX 4090)
- **Software**: PyTorch, CUDA, NCCL backend
- **Dataset**: ImageNet (ILSVRC2012) organized in ImageFolder format

### Environment Setup

```bash
# Install dependencies
bash tools/setup_environment.sh

# Verify environment
python tools/check_environment.py

# (Optional) If using Kaggle for dataset download, create kaggle.json from template
cp kaggle.json.template kaggle.json
# Then edit kaggle.json with your credentials
```

### Data Preparation

```bash
# Prepare ImageNet dataset
bash scripts/data_preparation/prepare_imagenet.sh

# Organize validation set
bash scripts/data_preparation/organize_val_fast.sh

# Verify data structure
ls -lh train/ val/
```

### Run Experiments

**Option 1: Automated (Recommended)**
```bash
# Run all experiments and generate analysis
bash scripts/analysis/run_all_experiments.sh
```

**Option 2: Manual Execution**
```bash
# Experiment A: Baseline DDP
torchrun --nproc_per_node=4 scripts/training/baseline_multi_card.py

# Experiment B: Manual All-Reduce
torchrun --nproc_per_node=4 scripts/training/all_reduce_train.py

# Experiment C: Parameter Server
torchrun --nproc_per_node=4 scripts/training/ps_train.py

# Generate analysis
python scripts/analysis/analyze_results.py
```

### View Results

```bash
# Check performance metrics
cat results/performance_report.txt

# View generated plots
ls results/plots/
```

## 📊 Training Configuration

- **Model**: ResNet50 (trained from scratch, no pretrained weights)
- **Dataset**: ImageNet (~1.28M training images, 50K validation images, 1000 classes)
- **Training Setup**:
  - Batch size: 64 per GPU
  - Epochs: 3
  - GPUs: 4 (single node)
  - DataLoader workers: 16
  - Backend: NCCL

## 🔬 Experimental Comparison

This project implements and compares three distributed training approaches:

### 1. Baseline DDP (`baseline_multi_card.py`)
- PyTorch native `DistributedDataParallel`
- Automatic gradient synchronization
- Standard reference implementation

### 2. Manual All-Reduce (`all_reduce_train.py`)
- Custom gradient synchronization using `dist.all_reduce()`
- Manual control over communication timing
- Demonstrates low-level collective operations

### 3. Parameter Server (`ps_train.py`)
- Synchronous PS architecture (Rank 0 = Parameter Server)
- Workers push gradients, PS aggregates and broadcasts
- Traditional distributed training paradigm

## 📈 Performance Metrics

All experiments output JSON files with:
- **Training throughput** (images/sec)
- **Total training time** (seconds)
- **Communication overhead** (%)
- **Loss convergence** (training & validation)
- **Accuracy** (top-1 validation accuracy)
- **Per-epoch statistics**

## 🎨 Visualizations

The `analyze_results.py` script generates 5 comparative plots:
1. Training throughput comparison
2. Total training time comparison
3. Loss convergence curves
4. Communication overhead comparison
5. Validation accuracy progression

## 📝 Key Features

- ✅ **No pretrained weights**: All models train from random initialization for fair comparison
- ✅ **Complete validation**: All scripts include validation loops with metrics
- ✅ **JSON output**: Structured results for easy analysis
- ✅ **Automated analysis**: One-command experiment execution and reporting
- ✅ **Resumable data prep**: Interrupted dataset extraction can be resumed
- ✅ **Git-friendly**: Large dataset files excluded via `.gitignore`

## 🛠️ Development

### Adding New Experiments

1. Create training script in `scripts/training/`
2. Output results JSON to `results/results_<name>.json`
3. Update `scripts/analysis/run_all_experiments.sh`
4. Update `scripts/analysis/analyze_results.py` to include new results

### Modifying Analysis

Edit `scripts/analysis/analyze_results.py` to customize:
- Plot styles and formatting
- Performance metrics calculation
- Report generation logic

## 📚 Documentation

- **Experiment Guide**: See `docs/EXPERIMENT_GUIDE.md` for detailed instructions
- **Environment Setup**: See `docs/ENVIRONMENT_SETUP.md` for configuration details
- **File Descriptions**: See `docs/FILES.md` for complete file listing
- **Chinese Documentation**: See `docs/README_CN.md` for Chinese version

## 🐛 Troubleshooting

### Dataset Issues
```bash
# Check data structure
find train/ -type d | head -20
find val/ -type d | head -20

# Verify image counts
find train/ -name "*.JPEG" | wc -l  # Should be ~1,281,167
find val/ -name "*.JPEG" | wc -l    # Should be 50,000
```

### CUDA/NCCL Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify GPU count
python -c "import torch; print(torch.cuda.device_count())"

# Check NCCL backend
python -c "import torch.distributed as dist; print(dist.is_nccl_available())"
```

### Results Missing
```bash
# Ensure results directory exists
mkdir -p results/plots

# Check for output files
ls -lh results/*.json
```

## 📄 License

This project is for educational purposes as part of AI Engineering coursework.

## 🙏 Acknowledgments

- PyTorch team for distributed training primitives
- ImageNet dataset from Stanford Vision Lab
- Course instructors and TAs

---

**Note**: This project compares distributed training methods on a single-node, multi-GPU setup. For multi-node experiments, adjust `torchrun` parameters accordingly.
