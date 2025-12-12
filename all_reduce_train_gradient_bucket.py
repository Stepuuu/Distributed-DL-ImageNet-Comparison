import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from tqdm import tqdm

class ManualDDP(nn.Module):
    def __init__(self, model, device, world_size, bucket_size=8):
        super(ManualDDP, self).__init__()
        self.model = model
        self.device = device
        self.world_size = world_size
        self.bucket_size = bucket_size
        self.grad_buckets = []  

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def backward_and_all_reduce(self, loss):
        
        loss.backward()
        
        for idx, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                bucket_idx = idx // self.bucket_size  
                if len(self.grad_buckets) <= bucket_idx:
                    self.grad_buckets.append([])
                self.grad_buckets[bucket_idx].append(param.grad.data)

        for bucket in self.grad_buckets:
            if len(bucket) >= self.bucket_size:
                for grad in bucket:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    grad /= self.world_size
                bucket.clear()  

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Manual All-Reduce and Gradient Bucketing')
    parser.add_argument('--epochs', default=1, type=int, help='Number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, help='Mini-batch size per process (default: 64)')
    parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--data-dir', default='./', type=str, help='Path to ImageNet dataset')
    parser.add_argument('--backend', default='nccl', type=str, help='Distributed backend (default: nccl)')
    parser.add_argument('--bucket-size', default=8, type=int, help='Size of gradient buckets')
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    num_classes = 1000
    model = models.resnet50(weights=None)
    local_weights_path = "./resnet50-0676ba61.pth"
    state_dict = torch.load(local_weights_path)
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda(local_rank)

    model = ManualDDP(model, device, world_size, bucket_size=args.bucket_size)


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    train_path = os.path.join(args.data_dir, 'train')
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_start_time = time.time()

        loop = tqdm(train_loader, desc=f"Train Phase", leave=False)
        total = 0
        correct = 0

        for images, target in loop:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            _, predicted = output.max(1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            optimizer.zero_grad()
            model.backward_and_all_reduce(loss)

            optimizer.step()

            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

        epoch_time = time.time() - epoch_start_time

        if rank == 0:
            print(f"Epoch [{epoch}] completed in {epoch_time:.2f}s")

if __name__ == '__main__':
    main()
