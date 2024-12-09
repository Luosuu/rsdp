from typing import List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh
# Reuse same method as much as possible
from torch.distributed.fsdp._fully_shard._fsdp_param_group import _get_param_module_infos
from ._rsdp_common import RSDPMeshInfo

class RSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype
    _reduce_dtype: Optional[torch.dtype]

    def __init__(
        self, 
        params: List[torch.nn.Parameter],
        modules: Tuple[nn.Module, ...],
        post_forward_mesh_info: Optional[RSDPMeshInfo],
        mesh_info: RSDPMeshInfo,
        device: torch.device,
        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]],
    ):
        self.modules = modules  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, modules)

        # Initialize sharded parameters using DTensor
        self.sharded_params = []
        for param in params:
            # Create sharded DTensor for each parameter
            sharded_param = self._create_sharded_param(param)
            self.sharded_params.append(sharded_param)

        # Buffers for ring communication
        self.recv_buffer = None
        self.current_shard_idx = self.rank

    def _create_sharded_param(self, param: torch.nn.Parameter) -> DTensor:
        """Create a sharded DTensor from a regular parameter"""
        # Shard along the first dimension by default
        sharding_spec = [self.world_size] + [1] * (param.dim() - 1)
        return DTensor.from_local(
            param,
            device_mesh=self.device_mesh,
            placements=sharding_spec,
        )

    def prepare_computation(self):
        """Prepare for ring-based computation"""
        # Allocate receive buffer for next shard
        buffer_size = self.sharded_params[0].size()
        self.recv_buffer = torch.empty(
            buffer_size,
            device=self.device,
            dtype=self.sharded_params[0].dtype
        )

    def exchange_shards(self):
        """Exchange shards in ring pattern"""
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size

        for param in self.sharded_params:
            # Get local tensor from DTensor for communication
            local_tensor = param._local_tensor

            # Asynchronous send/receive
            send_work = dist.isend(local_tensor, next_rank)
            recv_work = dist.irecv(self.recv_buffer, prev_rank)

            # Wait for communication to complete
            send_work.wait()
            recv_work.wait()

            # Update DTensor with received shard
            param._local_tensor.copy_(self.recv_buffer)

        # Update current shard index
        self.current_shard_idx = (self.current_shard_idx - 1) % self.world_size

    def compute_partial(self, forward_fn):
        """Compute using current shard"""
        # Perform computation on current local shards
        partial_outputs = []
        for param in self.sharded_params:
            # Get local tensor for computation
            local_data = param._local_tensor

            # Apply computation
            partial_output = forward_fn(local_data)
            partial_outputs.append(partial_output)

        return partial_outputs

    def reshard(self):
        """Return parameters to original sharding state"""
        # Exchange shards until we return to original configuration
        while self.current_shard_idx != self.rank:
            self.exchange_shards()

    def cleanup(self):
        """Clean up temporary buffers"""
        self.recv_buffer = None
