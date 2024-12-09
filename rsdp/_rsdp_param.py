from typing import List, Optional, Tuple, Callable, cast
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh
# Reuse same methods in FSDP
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo, _from_local_no_grad
from torch.distributed.fsdp._fully_shard._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState, FSDPParam, ParamModuleInfo

# TODO: Finish Implementation
class RSDPParam(FSDPParam):
    """
    Manages a parameter with Ring-Sharded Data Parallel (RSDP) applied.
    Key difference from FSDPParam: Uses ring-based communication pattern
    instead of all-gather for parameter reconstruction.
    """
    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        device: torch.device,
        shard_placement_fn: Optional[Callable],
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
        ):
        super().__init__(
            param=param,
            module_info=module_info,
            mesh_info=mesh_info,
            post_forward_mesh_info=None,  # RSDP doesn't need this
            device=device,
            shard_placement_fn=shard_placement_fn,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
        self._init_ring_attributes()

    def _init_ring_attributes(self):
        """Initialize ring-specific attributes"""
        self.world_size = self.mesh_info.mesh.size()
        self.rank = self.mesh_info.mesh.rank()
        self.current_shard_idx = self.rank
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1) % self.world_size
        self.recv_buffer = torch.empty_like(self._sharded_param_data)
        self.partial_results = []
        self.in_ring_computation = False

    def prepare_computation(self):
        """Prepare for ring-based computation"""
        if self.recv_buffer is None:
            # Allocate receive buffer for next shard
            self.recv_buffer = torch.empty_like(self._sharded_param_data)

    def exchange_shards(self):
        """Exchange shards in ring pattern"""
        if self.sharded_state != ShardedState.SHARDED:
            raise RuntimeError("Can only exchange shards in SHARDED state")

        # Send current shard to next rank
        send_work = dist.isend(
            self._sharded_param_data, 
            self.next_rank
        )

        # Receive new shard from previous rank
        recv_work = dist.irecv(
            self.recv_buffer,
            self.prev_rank
        )

        # Wait for communication to complete
        send_work.wait()
        recv_work.wait()

        # Update current shard
        self._sharded_param_data.copy_(self.recv_buffer)

        # Update current shard index
        self.current_shard_idx = (self.current_shard_idx - 1) % self.world_size

    def compute_partial(self, forward_fn):
        """Compute using current shard"""
        # Get local tensor for computation
        local_tensor = self._sharded_local_tensor

        # Apply computation
        partial_output = forward_fn(local_tensor)
        self.partial_results.append(partial_output)

        return partial_output

    def reset_to_original_shard(self):
        """Return to original shard configuration"""
        while self.current_shard_idx != self.rank:
            self.exchange_shards()

        # Clear partial results
        self.partial_results.clear()

    # Reuse other methods from FSDPParam
    def _init_sharded_param(self, param, device, shard_placement_fn):
        # Similar to FSDPParam but simplified for ring sharding
        pass

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast(DTensor, self.sharded_param)._local_tensor

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """Convert local tensor to DTensor with ring sharding spec"""
        return _from_local_no_grad(
            tensor,
            self._sharding_spec,
        )
