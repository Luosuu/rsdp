from typing import List, Optional, Tuple, Callable, Dict, NamedTuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh, _get_device_handle
# Reuse same method as much as possible
from torch.distributed.fsdp._fully_shard._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import _get_param_module_infos
from torch.distributed.fsdp._fsdp_common import (
    compiled_autograd_enabled,
    FSDPMeshInfo,
    HSDPMeshInfo,
    TrainingState,
)
from torch.utils.hooks import RemovableHandle
from ._rsdp_param import RSDPParam

_ModuleToHandleDict = Dict[nn.Module, RemovableHandle]  # for state dict

class RSDPCommContext:
    """Communication state shared across RSDP states/parameter groups."""

    def lazy_init(self, device: torch.device):
        self.device_handle = _get_device_handle(device.type)
        high_priority = -1

        # Ring communication streams
        # Stream for preparing data to be sent in ring communication
        self.ring_exchange_copy_in_stream = self.device_handle.Stream(
            priority=high_priority
        )
        # Stream for ring communication operations
        self.ring_exchange_stream = self.device_handle.Stream(
            priority=high_priority
        )
        # Stream for computation on received data
        self.ring_compute_stream = self.device_handle.Stream(
            priority=high_priority
        )
        # Stream for post-computation operations
        self.post_compute_stream = self.device_handle.Stream(
            priority=high_priority
        )

        # Ring communication states
        self.ring_exchange_state: Optional[RingExchangeState] = None
        self.ring_compute_state: Optional[RingComputeState] = None

        # For backward pass (opposite direction ring communication)
        self.backward_ring_state: Optional[BackwardRingState] = None

        # Post-forward order for explicit backward handling
        self.post_forward_order: List[RSDPParamGroup] = []

    def get_ring_exchange_streams(
        self, async_op: bool, training_state: TrainingState
    ) -> Tuple[torch.Stream, torch.Stream, torch.Stream]:
        if not async_op and training_state in (
            TrainingState.FORWARD,
            TrainingState.PRE_BACKWARD,
        ):
            # Use separate streams for pipelined ring communication
            return (
                self.ring_exchange_copy_in_stream,
                self.ring_exchange_stream,
                self.ring_compute_stream
            )
        current_stream = self.device_handle.current_stream()
        return current_stream, current_stream, current_stream


class RingExchangeState(NamedTuple):
    """State for ring-based parameter exchange"""
    send_tensor: torch.Tensor  # Tensor being sent to next rank
    recv_tensor: torch.Tensor  # Tensor received from previous rank
    send_event: torch.Event    # Event for send completion
    recv_event: torch.Event    # Event for receive completion
    current_shard_idx: int     # Current shard index in ring rotation


class RingComputeState(NamedTuple):
    """State for computation on received data"""
    compute_tensor: torch.Tensor  # Current tensor for computation
    partial_results: List[torch.Tensor]  # Accumulated partial results
    compute_event: torch.Event  # Event for computation completion


class BackwardRingState(NamedTuple):
    """State for backward pass ring communication"""
    grad_tensor: torch.Tensor  # Gradient tensor for ring exchange
    send_event: torch.Event    # Event for gradient send completion
    recv_event: torch.Event    # Event for gradient receive completion
    partial_grads: List[torch.Tensor]  # Partial gradient accumulation


class RSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype
    _reduce_dtype: Optional[torch.dtype]

    def __init__(
        self, 
        params: List[torch.nn.Parameter],
        modules: Tuple[nn.Module, ...],
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        mesh_info: FSDPMeshInfo,
        device: torch.device,
        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]],
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy,
    ):
        self.modules = modules  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, modules)

        self.rsdp_params = [
            RSDPParam(
                param,
                module_info,
                mesh_info,
                post_forward_mesh_info,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
            for param, module_info in zip(params, param_module_infos)
        ]

        # Buffers for ring communication
        self.mesh_info = mesh_info
        self.device = device
        self.device_handle = _get_device_handle(device.type)
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self._training_state = TrainingState.IDLE

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
        for param in self.params:
            param.prepare_computation()

    def exchange_shards(self):
        """Exchange shards in ring pattern"""
        for param in self.params:
            param.exchange_shards()

        # Update current shard index
        self.current_shard_idx = (self.current_shard_idx - 1) % self.world_size


    def compute_partial(self, forward_fn):
        """Compute using current shards"""
        partial_results = []
        for param in self.params:
            partial_result = param.compute_partial(forward_fn)
            partial_results.append(partial_result)
        return partial_results

    def reshard(self):
        """Return parameters to original sharding state"""
        # Exchange shards until we return to original configuration
        while self.current_shard_idx != self.rank:
            self.exchange_shards()

        # Reset params to original sharded state
        for param in self.params:
            param.to_sharded()

    def cleanup(self):
        """Clean up temporary buffers"""
        self.recv_buffer = None

    @property
    def sharded_params(self) -> List[nn.Parameter]:
        """Get list of sharded parameters"""
        return [param.sharded_param for param in self.params]

    @property
    def is_sharded(self) -> bool:
        """Check if all parameters are in sharded state"""
        return all(param.sharded_state == ShardedState.SHARDED for param in self.params)

    def _validate_state(self, expected_state: ShardedState):
        """Validate all parameters are in expected state"""
        for param in self.params:
            if param.sharded_state != expected_state:
                raise RuntimeError(
                    f"Expected all parameters to be in {expected_state} "
                    f"but found {param.sharded_state}"
                )

    def get_param_by_name(self, param_name: str) -> Optional[RSDPParam]:
        """Get RSDPParam by parameter name"""
        for param in self.params:
            if param._module_info.param_name == param_name:
                return param
        return None
