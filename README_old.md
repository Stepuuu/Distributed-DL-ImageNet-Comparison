# Prepare your dataset

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

mkdir train
tar xvf ILSVRC2012_img_train.tar -C ./train
chmod +x ./unzip.sh
./unzip.sh

mkdir val
tar xvf ILSVRC2012_img_val.tar -C ./val
cd val
chomod +x ./valprep.sh
./valprep.sh
```

# Prepare your pre-trained model and 

`wget https://download.pytorch.org/models/resnet50-0676ba61.pth`

# Run baseline single card

`python baseline_single_card.py`

# Run baseline multiple cards

`torchrun --nproc_per_node 4 baseline_multi_card.py`

# Run parameter-server script

`torchrun --nproc_per_node 4 ps_train.py`


# Run all_reduce script

```
export NCCL_ALGO=ring
torchrun --nproc_per_node=4 all_reduce_train.py
```

```
export NCCL_ALGO=tree 
torchrun --nproc_per_node 4 all_reduce_train.py
```

# Ablation Study 

## run bucket_reduction script
`torchrun --nproc_per_node 4 all_reduce_train_gradient_bucket.py`


## run bucket_reduction script
`torchrun --nproc_per_node 4 all_reduce_train_overlap.py`


## run with both techniques added 
`torchrun --nproc_per_node 4 all_reduce_optimize_ddp_v1.5.py`