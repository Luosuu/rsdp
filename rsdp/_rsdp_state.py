# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import torch
import torch.nn as nn
from torch._logging import warning_once
from torch.autograd import Variable
from torch.autograd.graph import _MultiHandle
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _to_kwargs
from torch.utils._pytree import tree_flatten, tree_map

from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import (
    _cast_fp_tensor,
    compiled_autograd_enabled,
    detect_compiled_autograd,
    TrainingState,
)
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup


#TODO: finish implementation
class RSDPState(_State):
    def __init__(self) -> None:
        return


def _get_module_rsdp_state(module: nn.Module) -> Optional[RSDPState]:
    state = _get_module_state(module)
    if isinstance(state, RSDPState):
        return state
    return None