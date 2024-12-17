import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor

import torch.nn as nn
from torch.distributed._tensor import (
    DeviceMesh,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from torch.distributed.device_mesh import init_device_mesh

WORLD_SIZE = torch.cuda.device_count()
ITER_TIME = 20

def print0(msg, rank):
    if rank == 0:
        print(msg)

# DDP replicates model weights
def demo_ddp_matmul():
    device_mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    rank = dist.get_rank()
    linear_w1 = torch.randn(1024, 1024)
    linear_w2 = torch.randn(1024, 1024)

    rowwise_placement = [Shard(0)]
    replicate_placement = [Replicate()]

    dist_w1 = distribute_tensor(linear_w1, device_mesh, replicate_placement)
    dist_w2 = distribute_tensor(linear_w2, device_mesh, replicate_placement) 

    input = torch.randn(2048, 1024)
    sharded_input = distribute_tensor(input, device_mesh, rowwise_placement)

    intermediate_input = torch.matmul(sharded_input.to_local(), dist_w1.to_local())
    final_output = torch.matmul(intermediate_input, dist_w2.to_local())

    return final_output

demo_ddp_matmul()

# TP replicates input
def demo_tp_matmul():
    device_mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    rank = dist.get_rank()
    linear_w1 = torch.randn(1024, 1024)
    linear_w2 = torch.randn(1024, 1024)

    rowwise_placement = [Shard(0)]
    colwise_placement=[Shard(1)]
    replicate_placement = [Replicate()]

    sharded_w1 = distribute_tensor(linear_w1, device_mesh, placements=colwise_placement)
    sharded_w2 = distribute_tensor(linear_w1, device_mesh, placements=rowwise_placement)

    input = torch.randn(1024, 1024)
    dist_input = distribute_tensor(input, device_mesh, replicate_placement)
    intermediate_input = torch.matmul(dist_input.to_local(), sharded_w1.to_local())
    # this is only half of the result, need to fetch another half from the other rank.
    final_output = torch.matmul(intermediate_input, sharded_w2.to_local())
    return final_output

demo_tp_matmul()

# FSDP replicates sub-module at computation
def demo_fsdp_matmul():
    device_mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    rank = dist.get_rank()
    linear_w1 = torch.randn(1024, 1024)
    linear_w2 = torch.randn(1024, 1024)

    rowwise_placement = [Shard(0)]

    sharded_w1 = distribute_tensor(linear_w1, device_mesh, placements=rowwise_placement)
    print0(f"FSDP: w1 global shape:{sharded_w1.shape}, local shape: {sharded_w1.to_local().shape}", rank)
    sharded_w2 = distribute_tensor(linear_w2, device_mesh, placements=rowwise_placement)

    input = torch.randn(2048, 1024)
    sharded_input = distribute_tensor(input, device_mesh, rowwise_placement)

    # before forward w1, restore w1 on each rank
    replicate_placement = [Replicate()]
    sharded_w1 = sharded_w1.redistribute(device_mesh, replicate_placement)
    print0(f"FSDP: before forward, w1 global shape:{sharded_w1.shape}, local shape: {sharded_w1.to_local().shape}", rank)
    # forward w1
    intermediate_input = torch.matmul(sharded_input.to_local(), sharded_w1.to_local())
    # reshard w1
    sharded_w1.redistribute(device_mesh, placements=rowwise_placement)

    # same for w2
    sharded_w2 = sharded_w2.redistribute(device_mesh, replicate_placement)
    final_output = torch.matmul(intermediate_input, sharded_w2.to_local())
    sharded_w2.redistribute(device_mesh, placements=rowwise_placement)

    return final_output

demo_fsdp_matmul()

# RSDP avoids any replicate
def demo_rsdp_matmul():
    device_mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    rank = dist.get_rank()
    linear_w1 = torch.randn(1024, 1024)
    linear_w2 = torch.randn(1024, 1024)
    
    rowwise_placement = [Shard(0)]
    colwise_placement=[Shard(1)]
    
    sharded_w1 = distribute_tensor(linear_w1, device_mesh, colwise_placement)
    sharded_w2 = distribute_tensor(linear_w2, device_mesh, colwise_placement)

    input = torch.randn(2048, 1024)
    sharded_input = distribute_tensor(input, device_mesh, rowwise_placement)

    # First partial compute on first layer
    partial00 = torch.matmul(sharded_input.to_local(), sharded_w1.to_local())
    print0(f"RSDP: First partial compute shape on first layer: {partial00.shape}", rank)
    # Redistribute first layer
    device_mesh_transit = DeviceMesh("cuda", [1, 0]) # reverse the mesh
    sharded_w1 = sharded_w1.redistribute(device_mesh_transit, colwise_placement)

    # Second partial compute
    partial01 = torch.matmul(sharded_input.to_local(), sharded_w1.to_local())
    print0(f"RSDP: Second partial compute shape on first layer: {partial01.shape}", rank)

    # Concat intermediate values
    intermediate_input = torch.concat([partial00, partial01], dim=1)

    # First partial compute on second layer
    partial10 = torch.matmul(intermediate_input, sharded_w2.to_local())
    print0(f"RSDP: First partial compute shape on second layer: {partial10.shape}", rank)
    # Redistribute second layer
    sharded_w2 = sharded_w2.redistribute(device_mesh_transit, colwise_placement)
    partial11 = torch.matmul(intermediate_input, sharded_w2.to_local())
    print0(f"RSDP: Second partial compute shape on second layer: {partial11.shape}", rank)

    # Concat for final output
    final_output = torch.concat([partial10, partial11], dim=1)

    return final_output

demo_rsdp_matmul()
    