
import torch
import torch.nn as nn
from torch.distributed.fsdp._fsdp_api import (
    MixedPrecisionPolicy, 
    OffloadPolicy,
)
from torch.distributed.tensor import DeviceMesh
from ._rsdp_shard import rsdp_shard

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




