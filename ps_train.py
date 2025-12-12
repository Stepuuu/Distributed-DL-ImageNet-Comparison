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


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch ImageNet Training with Parameter-Server"
    )
    parser.add_argument(
        "--epochs", default=3, type=int, help="Number of total epochs to run"
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Mini-batch size per process (default: 256)",
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
    is_ps = rank == 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    num_classes = 1000
    model = models.resnet50(weights=None)

    local_weights_path = "./resnet50-0676ba61.pth"
    state_dict = torch.load(local_weights_path)
    model.load_state_dict(state_dict)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda(local_rank)

    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    if is_ps:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if not is_ps:
        train_path = os.path.join(args.data_dir, "train")
        val_path = os.path.join(args.data_dir, "val")
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

        # 使用 DistributedSampler 确保每个 Worker 处理不同的数据部分
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=world_size - 1,  # 排除 PS 进程
            rank=rank - 1,  # PS 的 rank 为 0，需要减掉
        )
        val_sampler = torch.utils.data.SequentialSampler(valid_data)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=args.batch_size,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=args.workers,
        )

        if rank == 1:
            num_batches = len(train_loader)
            num_batches_tensor = torch.tensor([num_batches], device="cuda")
            dist.send(num_batches_tensor, dst=0)
    else:
        num_batches_tensor = torch.tensor([0], device="cuda")
        dist.recv(num_batches_tensor, src=1)
        num_batches = num_batches_tensor.item()

    if rank == 1:
        all_loss = []
        metrics = {
            "method": "Parameter Server (PS)",
            "world_size": world_size,
            "num_workers": world_size - 1,
            "batch_size": args.batch_size,
            "epochs": [],
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        print("=" * 80)
        print("Starting Parameter Server Training")
        print(f"World Size: {world_size}, PS: Rank 0, Workers: Rank 1-{world_size - 1}")
        print(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}")
        print("=" * 80)

    for epoch in range(args.epochs):
        total = 0
        correct = 0

        if rank == 1:
            print(f"\n{'=' * 80}")
            print(f"Epoch [{epoch + 1}/{args.epochs}]")
            print(f"{'=' * 80}")

        if not is_ps:
            train_sampler.set_epoch(epoch)
            model.train()
            epoch_start_time = time.time()

            batch_times = []
            comm_times_download = []
            comm_times_upload = []

            loop = tqdm(
                train_loader,
                desc=f"Train Phase Worker-{rank}",
                leave=False,
                disable=(rank != 1),
            )

            for batch_idx, (images, target) in enumerate(loop):
                batch_start = time.time()
                images = images.cuda(local_rank, non_blocking=True)
                target = target.cuda(local_rank, non_blocking=True)

                # 从 PS 接收最新的模型参数
                start_comm = time.time()
                for param in model.parameters():
                    dist.recv(param.data, src=0)
                comm_time_download = time.time() - start_comm
                comm_times_download.append(comm_time_download)

                output = model(images)
                loss = criterion(output, target)

                _, predicted = output.max(1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

                model.zero_grad()
                loss.backward()

                # 发送梯度给 PS
                start_comm = time.time()
                for param in model.parameters():
                    dist.send(param.grad.data, dst=0)
                comm_time_upload = time.time() - start_comm
                comm_times_upload.append(comm_time_upload)

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                if rank == 1:
                    throughput = images.size(0) / batch_time if batch_time > 0 else 0
                    loop.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{100.0 * correct / total:.2f}%",
                        throughput=f"{throughput:.1f} img/s",
                    )

                if rank == 1:
                    all_loss.append(loss.item())

            # Training epoch完成后统计
            epoch_time = time.time() - epoch_start_time
            epoch_accuracy = 100.0 * correct / total if total > 0 else 0
            avg_loss = (
                sum(all_loss[-len(train_loader) :]) / len(train_loader)
                if all_loss
                else 0
            )
            throughput = total / epoch_time if epoch_time > 0 else 0
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            avg_comm_download = (
                sum(comm_times_download) / len(comm_times_download)
                if comm_times_download
                else 0
            )
            avg_comm_upload = (
                sum(comm_times_upload) / len(comm_times_upload)
                if comm_times_upload
                else 0
            )

            if rank == 1:
                print(
                    f"\n[Epoch {epoch + 1}] Train Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
                )
                print(
                    f"[Epoch {epoch + 1}] Train Time: {epoch_time:.2f}s, Throughput: {throughput:.2f} img/s"
                )
                print(
                    f"[Epoch {epoch + 1}] Avg Batch Time: {avg_batch_time * 1000:.2f}ms"
                )
                print(
                    f"[Epoch {epoch + 1}] Avg Comm Time (Pull): {avg_comm_download * 1000:.2f}ms, (Push): {avg_comm_upload * 1000:.2f}ms"
                )

            ## testing
            model.eval()
            running_loss = 0.0
            correct_val = 0
            total_val = 0

            loop = tqdm(val_loader, desc="Test Phase", leave=False, disable=(rank != 1))

            for images, labels in loop:
                images, labels = images.to("cuda"), labels.to("cuda")

                with torch.set_grad_enabled(False):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                if rank == 1:
                    loop.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{100.0 * correct_val / total_val:.2f}%",
                    )

            epoch_val_loss = running_loss / total_val if total_val > 0 else 0
            epoch_val_accuracy = 100.0 * correct_val / total_val if total_val > 0 else 0

            if rank == 1:
                print(
                    f"[Epoch {epoch + 1}] Val Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_accuracy:.2f}%"
                )

                metrics["epochs"].append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "train_accuracy": epoch_accuracy,
                        "train_time": epoch_time,
                        "train_throughput": throughput,
                        "avg_batch_time": avg_batch_time,
                        "avg_comm_time_pull": avg_comm_download,
                        "avg_comm_time_push": avg_comm_upload,
                        "val_loss": epoch_val_loss,
                        "val_accuracy": epoch_val_accuracy,
                    }
                )

        else:
            # 参数服务器接收梯度，更新参数，并发送更新后的参数
            model.train()
            num_workers = world_size - 1
            epoch_start_time = time.time()

            ps_comm_times_send = []
            ps_comm_times_recv = []
            ps_batch_times = []

            if is_ps:
                print(
                    f"[PS] Epoch [{epoch + 1}/{args.epochs}] - Processing {num_batches} batches..."
                )

            for batch_idx in range(num_batches):
                batch_start = time.time()

                # 向 Workers 发送最新的模型参数 (Broadcast)
                start_comm = time.time()
                for param in model.parameters():
                    for worker_rank in range(1, world_size):
                        dist.send(param.data, dst=worker_rank)
                comm_time_send = time.time() - start_comm
                ps_comm_times_send.append(comm_time_send)

                # 接收来自 Workers 的梯度并累加 (Reduce)
                start_comm = time.time()
                for param in model.parameters():
                    grad_data = torch.zeros_like(param.data)
                    for worker_rank in range(1, world_size):
                        worker_grad = torch.zeros_like(param.data)
                        dist.recv(worker_grad, src=worker_rank)
                        grad_data += worker_grad
                    # 计算平均梯度
                    grad_data /= num_workers
                    # 更新参数
                    param.grad = grad_data
                comm_time_recv = time.time() - start_comm
                ps_comm_times_recv.append(comm_time_recv)

                optimizer.step()
                optimizer.zero_grad()

                batch_time = time.time() - batch_start
                ps_batch_times.append(batch_time)

            epoch_time = time.time() - epoch_start_time
            avg_ps_batch_time = (
                sum(ps_batch_times) / len(ps_batch_times) if ps_batch_times else 0
            )
            avg_ps_comm_send = (
                sum(ps_comm_times_send) / len(ps_comm_times_send)
                if ps_comm_times_send
                else 0
            )
            avg_ps_comm_recv = (
                sum(ps_comm_times_recv) / len(ps_comm_times_recv)
                if ps_comm_times_recv
                else 0
            )

            if is_ps:
                print(f"[PS] Epoch [{epoch + 1}] Completed in {epoch_time:.2f}s")
                print(f"[PS] Avg Batch Time: {avg_ps_batch_time * 1000:.2f}ms")
                print(
                    f"[PS] Avg Comm Time (Send Params): {avg_ps_comm_send * 1000:.2f}ms, (Recv Grads): {avg_ps_comm_recv * 1000:.2f}ms"
                )

    if rank == 1:
        metrics["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["all_losses"] = all_loss

        # Calculate average metrics
        avg_train_throughput = sum(
            e["train_throughput"] for e in metrics["epochs"]
        ) / len(metrics["epochs"])
        avg_train_time = sum(e["train_time"] for e in metrics["epochs"]) / len(
            metrics["epochs"]
        )
        avg_comm_pull = sum(e["avg_comm_time_pull"] for e in metrics["epochs"]) / len(
            metrics["epochs"]
        )
        avg_comm_push = sum(e["avg_comm_time_push"] for e in metrics["epochs"]) / len(
            metrics["epochs"]
        )
        best_val_accuracy = max(e["val_accuracy"] for e in metrics["epochs"])

        metrics["summary"] = {
            "avg_train_throughput": avg_train_throughput,
            "avg_train_time_per_epoch": avg_train_time,
            "avg_comm_time_pull": avg_comm_pull,
            "avg_comm_time_push": avg_comm_push,
            "best_val_accuracy": best_val_accuracy,
        }

        with open("./results_ps.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 80)
        print("Training Completed!")
        print(f"Average Throughput: {avg_train_throughput:.2f} img/s")
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
        print(
            f"Avg Communication Time - Pull: {avg_comm_pull * 1000:.2f}ms, Push: {avg_comm_push * 1000:.2f}ms"
        )
        print("Results saved to: results_ps.json")
        print("=" * 80)


if __name__ == "__main__":
    main()
