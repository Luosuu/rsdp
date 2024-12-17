# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DTensor,
    Shard,
    Replicate,
    distribute_tensor,
    distribute_module,
    DeviceMesh,
    init_device_mesh,
)

import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input) + self.fc2(input))


def shard_params(mod_name, mod, mesh):
    col_linear_placement = [Shard(0)]
    # shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, col_linear_placement)
            )
            mod.register_parameter(name, dist_param)


world_size = torch.cuda.device_count()
mesh = init_device_mesh("cuda", (world_size,))
sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)


def DDP(user_model: nn.Module, mesh: DeviceMesh):
    def input_fn(input: torch.Tensor):
        # input is conceptually sharded on batch dim 0
        if not isinstance(input, DTensor):
            # inputs are torch.Tensor on each rank from data loader, convert to distributed tensor.
            return DTensor.from_local(input, mesh, [Shard(0)])
        else:
            return input

    # by default replicate the module across the mesh
    distribute_module(user_model, device_mesh=mesh, input_fn=input_fn)
    return user_model


def FSDP(user_model: nn.Module):
    num_devices = torch.cuda.device_count()
    device_mesh = DeviceMesh(range(num_devices))
    # shard on dim 0
    shard_spec = [Shard(0)]
    replicate_spec = [Replicate()]

    # make entire model sharded across devices (fully sharded)
    sharded_model = distribute_module(user_model, device_mesh, placements=shard_spec)

    def before_forward_hook(module, input):
        # all_gather parameter for the module, simply transform fully sharded to replicate
        # from torch.distributed.fsdp._fully_shard.
        torch.distributed.fsdp.redistributed(
            module, device_mesh, placements=replicate_spec
        )

        # algorithm to find reasonable all_gather/reduce_scatter points across module tree
        # here we just list a example to simply install all_gather to first submodule
        sharded_model.submodules[0].register_forward_pre_hook(before_forward_hook)

    return sharded_model


def print0(msg, rank):
    if rank == 0:
        print(msg)



def demo_tp(world_size):
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native APIs.
    """
    rank = dist.get_rank()
    print0("Create a sharding plan based on the given world_size", rank)
    # create model and move it to GPU with id rank
    model = MyModule()
    # Create a optimizer for the parallelized module.
    LR = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    print0("Parallelize the module based on the given Parallel Style", rank)
    # Parallelize the module based on the given Parallel Style.
    tp_mesh = init_device_mesh("cuda", (world_size,))
    model = parallelize_module(
        model, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()}
    )

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for i in range(20):
        inp = torch.rand(20, 10)
        output = model(inp)
        print0(f"FWD Step: iter {i}", rank)
        output.sum().backward()
        print0(f"BWD Step: iter {i}", rank)
        optimizer.step()
        print0(f"Optimization Step: iter {i}", rank)

    print0("Training finished", rank)
