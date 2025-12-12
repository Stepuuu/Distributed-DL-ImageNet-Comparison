import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import itertools
from torchvision import models, datasets, transforms
from tqdm import tqdm

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.cuda._utils import _get_device_index

def _find_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class ManualDDP(Module):
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None, bucket_cap_mb=25
                ):
        
        super(ManualDDP, self).__init__()
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))

        if output_device is None:
            output_device = device_ids[0]
        self.output_device = _get_device_index(output_device, True)

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.dim = dim
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True

        MB = 1024 * 1024
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * MB)
        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * MB)
        # Sync params and buffers
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._distributed_broadcast_coalesced(
                module_states,
                self.broadcast_bucket_size)
        
        self._ddp_init_helper()

    def _build_debug_param_to_name_mapping(self, parameters):
        # pass to reducer
        param_to_param_index = {parameters[i]: i for i in range(len(parameters))}
        param_index_to_param_fqn = {}
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{param_name}"
                if param.requires_grad:
                    param_index = param_to_param_index[param]
                    param_index_to_param_fqn[param_index] = fqn

        return param_index_to_param_fqn

    def _build_params_for_reducer(self):
        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = [
            (module, parameter)
            for _, module in self.module.named_modules()
            for parameter in [
                param for _, param in module.named_parameters(recurse=False) if param.requires_grad
            ]
        ]
        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        modules_and_parameters = [
            (m, p)
            for m, p in modules_and_parameters if p not in memo and not memo.add(p)
        ]
        parameters = [parameter for _, parameter in modules_and_parameters]

        self._assign_modules_buffers()

        return parameters, [] # resnet50 do not have sparse gradient
    
    def _assign_modules_buffers(self):
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
        ]
        self.modules_buffers = [
            buffer for (buffer, _) in named_module_buffers
        ]
        self.named_module_buffers = {
            buffer_name: buffer for (buffer, buffer_name) in named_module_buffers
        }
    
    def _ddp_init_helper(self):
        """
        (1) replicating the module from device[0] to the other devices
        (2) bucketing the parameters for reductions
        (3) resetting the bucketing states
        (4) registering the grad hooks
        """
        parameters, expect_sparse_gradient = self._build_params_for_reducer()

        (bucket_indices, per_bucket_size_limits)= dist._compute_bucket_assignment_by_size(
            parameters,
            [1024 * 1024, self.bucket_bytes_cap],
            expect_sparse_gradient)

        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            list(reversed(per_bucket_size_limits)),
            self.process_group,
            expect_sparse_gradient,
            self.bucket_bytes_cap,
            False,
            False,
            param_to_name_mapping,
        )

    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            with torch.no_grad():
                if self.broadcast_buffers and len(self.modules_buffers[0]) > 0:
                    # Synchronize buffers across processes. Authoritative rank is 0. 
                    self._distributed_broadcast_coalesced(
                        self.modules_buffers,
                        self.broadcast_bucket_size,
                        authoritative_rank=0)

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            output = self.module(*inputs[0], **kwargs[0])
        else:
            output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train(self, mode=True):
        super().train(mode)
        return self

    def _distributed_broadcast_coalesced(self, tensors, buffer_size, authoritative_rank=0):
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )
                    
def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Manual All-Reduce')
    parser.add_argument('--epochs', default=3, type=int, help='Number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, help='Mini-batch size per process (default: 64)')
    parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--data-dir', default='./', type=str, help='Path to ImageNet dataset')
    parser.add_argument('--backend', default='nccl', type=str, help='Distributed backend (default: nccl)')
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

    model = ManualDDP(model, device_ids=[rank], output_device=rank)

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

            # Forward pass
            output = model(images)
            loss = criterion(output, target)

            _, predicted = output.max(1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

        epoch_time = time.time() - epoch_start_time

        if rank == 0:
            print(f"Epoch [{epoch}] completed in {epoch_time:.2f}s")

if __name__ == '__main__':
    main()
