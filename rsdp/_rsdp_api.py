from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    NoReturn,
    Optional,
    Type,
    Union,
)
from dataclasses import dataclass
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed._composable import contract
import torch.distributed as dist
from torch.distributed.utils import _get_root_modules
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from torch.distributed.fsdp._fsdp_api import (
    MixedPrecisionPolicy, 
    OffloadPolicy,
)
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor, distribute_module, init_device_mesh

# based on https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_common.py#L60
@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: Optional[int] = None
    replicate_mesh_dim: Optional[int] = None

    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )

@dataclass
class RSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()

# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L336
def apply_rsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    tp_enabled: bool,
    pp_enabled: bool,
    cpu_offload: bool = False,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    rsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        rsdp_config["offload_policy"] = OffloadPolicy()
    
    for layer_id, transformer_block in model.layers.items():
        if pp_enabled:
            # For PP, do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = False
        else:
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
        rsdp_shard(
            transformer_block,
            **rsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    rsdp_shard(model, **rsdp_config, reshard_after_forward=not pp_enabled)

# based on https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L51
# TODO: RSDPState
# TODO: RSDPParamGroup
# TODO: RSDPModule
# TODO: RSDP Class
# TODO: Ensure torch contractor compatible here
@contract(state_cls=RSDPState)
def rsdp_shard(
    module: Union[nn.Module, List[nn.Module]],
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
):
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            raise ValueError(
                f"fully_shard does not support containers that do not implement forward: {module}"
            )
    
    mesh = mesh or _init_default_fully_shard_mesh()
    mesh_info = RSDPMeshInfo(mesh, shard_mesh_dim=0)
    device = _get_device_from_mesh(mesh)

    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )

    state = rdt_shard.state(modules[0])
    state.init(modules, device, mp_policy)

    managed_modules = _get_managed_modules(modules)
    params, buffers = _get_managed_states(managed_modules)
    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = RSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_rsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"RSDP{cls.__name__}", (RSDPModule, cls), dct)
            cls_to_rsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module

def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "RSDP does not support deepcopy. Please use state dict for serialization."
    )


