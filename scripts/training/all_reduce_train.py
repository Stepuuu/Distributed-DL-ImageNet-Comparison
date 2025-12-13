import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from tqdm import tqdm


class ManualDDP(nn.Module):
    def __init__(self, model, device, world_size):
        super(ManualDDP, self).__init__()
        self.model = model
        self.device = device
        self.world_size = world_size

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def backward_and_all_reduce(self, loss):
        loss.backward()
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch ImageNet Training with Manual All-Reduce"
    )
    parser.add_argument(
        "--epochs", default=3, type=int, help="Number of total epochs to run"
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Mini-batch size per process (default: 64)",
    )
    parser.add_argument(
        "--workers",
        default=16,
        type=int,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--data-dir", default="./", type=str, help="Path to ImageNet dataset"
    )
    parser.add_argument(
        "--backend",
        default="nccl",
        type=str,
        help="Distributed backend (default: nccl)",
    )
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    num_classes = 1000
    # 从头训练，不使用预训练权重
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda(local_rank)

    model = ManualDDP(model, device, world_size)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    train_path = os.path.join(args.data_dir, "train")
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer,
        ]
    )
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    # 添加验证集
    val_path = os.path.join(args.data_dir, "val")
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer,
        ]
    )
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    if rank == 0:
        metrics = {
            "method": "Manual All-Reduce",
            "world_size": world_size,
            "batch_size": args.batch_size,
            "epochs": [],
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        print("=" * 80)
        print("Starting Manual All-Reduce Training")
        print(
            f"World Size: {world_size}, Batch Size: {args.batch_size}, Epochs: {args.epochs}"
        )
        print("=" * 80)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_start_time = time.time()

        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"Epoch [{epoch + 1}/{args.epochs}]")
            print(f"{'=' * 80}")

        loop = tqdm(train_loader, desc="Train Phase", leave=False, disable=(rank != 0))
        total = 0
        correct = 0
        all_losses = []
        batch_times = []
        comm_times = []

        for batch_idx, (images, target) in enumerate(loop):
            batch_start = time.time()
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            _, predicted = output.max(1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            all_losses.append(loss.item())

            optimizer.zero_grad()

            comm_start = time.time()
            model.backward_and_all_reduce(loss)
            comm_time = time.time() - comm_start
            comm_times.append(comm_time)

            optimizer.step()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if rank == 0:
                throughput = images.size(0) / batch_time if batch_time > 0 else 0
                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100.0 * correct / total:.2f}%",
                    throughput=f"{throughput:.1f} img/s",
                )

        epoch_time = time.time() - epoch_start_time
        throughput = total / epoch_time if epoch_time > 0 else 0
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_comm_time = sum(comm_times) / len(comm_times) if comm_times else 0
        epoch_accuracy = 100.0 * correct / total
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0

        if rank == 0:
            print(
                f"\n[Epoch {epoch + 1}] Train Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            )
            print(
                f"[Epoch {epoch + 1}] Epoch Time: {epoch_time:.2f}s, Throughput: {throughput:.2f} img/s"
            )
            print(
                f"[Epoch {epoch + 1}] Avg Batch Time: {avg_batch_time * 1000:.2f}ms, Avg Comm Time: {avg_comm_time * 1000:.2f}ms"
            )

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_loop = tqdm(
                    val_loader, desc="Validation", leave=False, disable=(rank != 0)
                )
                for images, labels in val_loop:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0
            val_loss = val_loss / val_total if val_total > 0 else 0

            print(
                f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

            metrics["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_accuracy": epoch_accuracy,
                    "train_time": epoch_time,
                    "train_throughput": throughput,
                    "avg_batch_time": avg_batch_time,
                    "avg_comm_time": avg_comm_time,
                    "all_losses": all_losses,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

    if rank == 0:
        metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate average metrics
        avg_train_throughput = sum(
            e["train_throughput"] for e in metrics["epochs"]
        ) / len(metrics["epochs"])
        avg_train_time = sum(e["train_time"] for e in metrics["epochs"]) / len(
            metrics["epochs"]
        )
        final_accuracy = metrics["epochs"][-1]["train_accuracy"]

        best_val_accuracy = max(e["val_accuracy"] for e in metrics["epochs"])
        final_val_accuracy = metrics["epochs"][-1]["val_accuracy"]

        metrics["summary"] = {
            "avg_train_throughput": avg_train_throughput,
            "avg_train_time_per_epoch": avg_train_time,
            "final_train_accuracy": final_accuracy,
            "best_val_accuracy": best_val_accuracy,
            "final_val_accuracy": final_val_accuracy,
        }

        # 创建results目录
        os.makedirs("./results", exist_ok=True)

        with open("./results/results_all_reduce.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Average Throughput: {avg_train_throughput:.2f} img/s")
        print(f"Final Training Accuracy: {final_accuracy:.2f}%")
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
        print("Results saved to: results/results_all_reduce.json")
        print("=" * 80)


if __name__ == "__main__":
    main()
