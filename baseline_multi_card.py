import json
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
from tqdm import tqdm


def load_data(batch_size, rank, world_size, path="./"):
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")

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
    valid_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalizer]
    )

    train_data = datasets.ImageFolder(root=train_path, transform=train_transform)
    valid_data = datasets.ImageFolder(root=val_path, transform=valid_transform)

    train_sampler = torch.utils.data.DistributedSampler(
        train_data, num_replicas=world_size, rank=rank
    )
    val_sampler = torch.utils.data.SequentialSampler(valid_data)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=16,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=16,
    )

    return train_loader, val_loader


def run_epoch(
    model, dataloader, criterion, optimizer=None, device="cpu", phase="train", rank=0
):
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    batch_times = []

    loop = tqdm(
        dataloader, desc=f"{phase.capitalize()} Phase", leave=False, disable=(rank != 0)
    )
    all_train_loss = []

    for batch_idx, (images, labels) in enumerate(loop):
        batch_start = time.time()
        images, labels = images.to(device), labels.to(device)

        if phase == "train":
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if phase == "train":
                loss.backward()
                optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_train_loss.append(loss.item())

        if rank == 0:
            throughput = images.size(0) / batch_time if batch_time > 0 else 0
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
                throughput=f"{throughput:.1f} img/s",
            )

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    throughput = total / epoch_time if epoch_time > 0 else 0

    return (
        epoch_loss,
        epoch_accuracy,
        all_train_loss,
        epoch_time,
        throughput,
        avg_batch_time,
    )


def train(batch_size, num_epochs, path="./"):
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    train_loader, val_loader = load_data(batch_size, rank, world_size, path)

    num_classes = 1000
    model = models.resnet50(weights=None)
    local_weights_path = "./resnet50-0676ba61.pth"
    state_dict = torch.load(local_weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    best_accuracy = 0.0
    save_path = "./saved_models"
    os.makedirs(save_path, exist_ok=True)

    if rank == 0:
        all_loss = []
        metrics = {
            "method": "Baseline DDP (PyTorch)",
            "world_size": world_size,
            "batch_size": batch_size,
            "epochs": [],
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        print("=" * 80)
        print("Starting Baseline DDP Training")
        print(
            f"World Size: {world_size}, Batch Size: {batch_size}, Epochs: {num_epochs}"
        )
        print("=" * 80)

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"{'=' * 80}")

        (
            train_loss,
            train_accuracy,
            all_train_loss,
            train_time,
            train_throughput,
            avg_batch_time,
        ) = run_epoch(
            model, train_loader, criterion, optimizer, device, phase="train", rank=rank
        )

        if rank == 0:
            all_loss.extend(all_train_loss)
            print(
                f"\n[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
            )
            print(
                f"[Epoch {epoch + 1}] Train Time: {train_time:.2f}s, Throughput: {train_throughput:.2f} img/s"
            )
            print(f"[Epoch {epoch + 1}] Avg Batch Time: {avg_batch_time * 1000:.2f}ms")

        val_loss, val_accuracy, _, val_time, val_throughput, _ = run_epoch(
            model, val_loader, criterion, device=device, phase="validate", rank=rank
        )

        if rank == 0:
            print(
                f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%"
            )
            print(f"[Epoch {epoch + 1}] Val Time: {val_time:.2f}s")

            metrics["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_time": train_time,
                    "train_throughput": train_throughput,
                    "avg_batch_time": avg_batch_time,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

        scheduler.step()

        if rank == 0 and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            print(
                f"[Epoch {epoch + 1}] New best model saved! Accuracy: {val_accuracy:.2f}%"
            )

    if rank == 0:
        metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["all_losses"] = all_loss
        metrics["best_accuracy"] = best_accuracy

        # Calculate average metrics
        avg_train_throughput = sum(
            e["train_throughput"] for e in metrics["epochs"]
        ) / len(metrics["epochs"])
        avg_train_time = sum(e["train_time"] for e in metrics["epochs"]) / len(
            metrics["epochs"]
        )

        metrics["summary"] = {
            "avg_train_throughput": avg_train_throughput,
            "avg_train_time_per_epoch": avg_train_time,
            "best_val_accuracy": best_accuracy,
        }

        # 创建results目录
        os.makedirs("./results", exist_ok=True)

        with open("./results/results_baseline_ddp.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Average Throughput: {avg_train_throughput:.2f} img/s")
        print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
        print("Results saved to: results/results_baseline_ddp.json")
        print("=" * 80)


if __name__ == "__main__":
    train(batch_size=64, num_epochs=3)
