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
from ._rsdp_common import RSDPMeshInfo
from ._rsdp_state import RSDPState, _get_module_rsdp_state

__all__ = [
    "fully_shard",
    "RSDPModule",
    "UnshardHandle",
    "register_fsdp_forward_method",
]


cls_to_rsdp_cls: Dict[Type, Type] = {}

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

    state = rsdp_shard.state(modules[0])
    state.init(modules, device, mp_policy)

    managed_modules = _get_managed_modules(modules)
    params, buffers = _get_managed_states(managed_modules)
    _move_states_to_device(params, buffers, device)
    if params:
        state._rsdp_param_group = RSDPParamGroup(
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

    # Place RSDP leftmost for highest priority in the method resolution order
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

class RSDPModule:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the RSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `FSDP<...>` class
        # and index 1 is the `RSDPModule` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        """
        Reshards the module's parameters, freeing the unsharded parameters if
        they are allocated and registering the sharded parameters to the
        module. This method is *not* recursive.
        """
        state = self._get_rsdp_state()
        if rsdp_param_group := state._rsdp_param_group:
            rsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> Optional["UnshardHandle"]:
        """
        Unshards the module's parameters by allocating memory and all-gathering
        the parameters. This method is *not* recursive. The unshard follows the
        :class:`MixedPrecisionPolicy`, so it will all-gather following
        ``param_dtype`` if set.

        Args:
            async_op (bool): If ``True``, then returns a :class:`UnshardHandle`
                that has a :meth:`wait` method to wait on the unshard op. If
                ``False``, then returns ``None`` and waits on the handle inside
                this function.

        .. note:: If ``async_op=True``, then FSDP will wait on the pending
            unshard in the module's pre-forward for the user. The user only
            needs to call :meth:`wait` explicitly if the wait should happen
            before pre-forward.
        """
        state = self._get_rsdp_state()
        rsdp_param_group = state._rsdp_param_group
        if rsdp_param_group is not None:
            rsdp_param_group.lazy_init()
            rsdp_param_group.unshard(async_op=async_op)
        handle = _UnshardHandleImpl(rsdp_param_group)
        if async_op:
            return handle
        handle.wait()
        return None

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one. On the last backward,
        FSDP waits on pending gradient reduction and clears internal data
        data structures for backward prefetching. This can be useful for
        microbatching.
        """
        state = self._get_rsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation *without communication*. For HSDP, this controls
        both reduce-scatter and all-reduce together.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, RSDPModule):
                state = module._get_rsdp_state()
                if rsdp_param_group := state._rsdp_param_group:
                    rsdp_param_group.reduce_grads = requires_gradient_sync
                    rsdp_param_group.all_reduce_grads = requires_gradient_sync

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should all-reduce gradients. This can be used to
        implement gradient accumulation with only reduce-scatter but not
        all-reduce for HSDP.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, RSDPModule):
                state = module._get_rsdp_state()
                if rsdp_param_group := state._rsdp_param_group:
                    rsdp_param_group.all_reduce_grads = requires_all_reduce

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication since the unsharded parameters do not need to be
        re-all-gathered before the next forward.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all FSDP submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, RSDPModule):
                state = module._get_rsdp_state()
                if rsdp_param_group := state._rsdp_param_group:
                    rsdp_param_group.reshard_after_backward = reshard_after_backward

    def set_modules_to_forward_prefetch(self, modules: List["RSDPModule"]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in forward. The prefetching runs after this
        module's all-gather copy-out.

        Passing a singleton list containing the next FSDP module gives the same
        all-gather overlap behavior as the default overlap behavior, except the
        prefetched all-gather is issued earlier from the CPU. Passing a list
        with at least length two is required for more aggressive overlap and
        will use more reserved memory.

        Args:
            modules (List[RSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_rsdp_modules(modules)
        self._get_rsdp_state()._states_to_forward_prefetch = [
            module._get_rsdp_state() for module in modules
        ]

    def set_modules_to_backward_prefetch(self, modules: List["RSDPModule"]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in backward. This overrides the default backward
        pretching implementation that prefetches the next FSDP module based on
        the reverse post-forward order.

        Passing a singleton list containing the previous FSDP module gives the
        same all-gather overlap behavior as the default overlap behavior.
        Passing a list with at least length two is required for more aggressive
        overlap and will use more reserved memory.

        Args:
            modules (List[RSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_rsdp_modules(modules)
        self._get_rsdp_state()._states_to_backward_prefetch = [
            module._get_rsdp_state() for module in modules
        ]

    def set_post_optim_event(self, event: torch.Event) -> None:
        """
        Sets a post-optimizer-step event for the root FSDP module to wait the
        all-gather streams on.

        By default, the root FSDP module waits the all-gather streams on the
        current stream to ensure that the optimizer step has finished before
        all-gathering. However, this may introduce false dependencies if
        there is unrelated computation after the optimizer step. This API
        allows the user to provide their own event to wait on. After the root
        waits on the event, the event is discarded, so this API should be
        called with a new event each iteration.

        Args:
            event (torch.Event): Event recorded after the optimizer step
                to wait all-gather streams on.
        """
        self._get_rsdp_state()._state_ctx.post_optim_event = event

    def set_reduce_scatter_divide_factor(self, factor: float) -> None:
        """
        Sets a custom divide factor for the reduce-scatter. This becomes a
        custom reduce op using NCCL's PreMulSum, which allows multiplying by
        the factor before reduction.

        Args:
            factor (float): Custom divide factor.
        """
        state = self._get_rsdp_state()
        if (rsdp_param_group := state._rsdp_param_group) is not None:
            mul_factor = 1.0 / float(factor)
            reduce_op = torch.distributed._make_nccl_premul_sum(mul_factor)
            rsdp_param_group.reduce_scatter_reduce_op = reduce_op

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        """
        Sets whether the FSDP module's parameters need to be unsharded in
        backward. This can be used in expert cases when the user knows that all
        parameters in this FSDP module's parameter group are not needed for
        backward computation (e.g. embedding).
        """
        state = self._get_rsdp_state()
        if (rsdp_param_group := state._rsdp_param_group) is not None:
            rsdp_param_group.unshard_in_backward = unshard_in_backward

    def _set_unshard_async_op(self, async_op: bool):
        """
        Sets whether to use ``async_op=True`` or ``False`` for the pre-forward
        and pre-backward unshard op. This defaults to ``False`` but can be set
        to ``True`` with this method.

        Setting this to ``True`` allows the all-gather allocations to happen in
        the default stream, avoiding inter-stream memory fragmentation.
        However, you must use explicit prefetching (e.g. via :meth:`unshard`)
        in forward to still get overlap, and the pre-all-gather ops like dtype
        casting and copy-in will not overlap with compute.
        """
        self_module = cast(nn.Module, self)
        for module in self_module.modules():
            if isinstance(module, RSDPModule):
                state = module._get_rsdp_state()
                if rsdp_param_group := state._rsdp_param_group:
                    rsdp_param_group.unshard_async_op = async_op

    def _get_rsdp_state(self) -> RSDPState:
        if (state := _get_module_rsdp_state(cast(nn.Module, self))) is None:
            raise AssertionError(f"No RSDP state found on {self}")
        return state

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        # Reshard to ensure that sharded parameters are registered
        self.reshard()
        ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
        state = self._get_rsdp_state()
        if not (rsdp_param_group := state._rsdp_param_group):
            return ret
        # TODO: Remove this padding logic once DTensor pads the local tensor:
        # https://github.com/pytorch/pytorch/issues/113045
        with torch.no_grad():
            for rsdp_param in rsdp_param_group.rsdp_params:
                rsdp_param.reset_sharded_param()
        return ret

class UnshardHandle:
    """
    A handle to wait on a :meth:`FSDPModule.unshard` op.
    """

    def wait(self) -> None:
        """
        Waits on the unshard op. This ensures that the current stream can use
        the unsharded parameters, which are now registered to the module.
        """
        return


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_group: Optional[RSDPParamGroup]):
        self._fsdp_param_group = fsdp_param_group

    def wait(self):
        if self._fsdp_param_group is not None:
            self._fsdp_param_group.wait_for_unshard()
            # Avoid keeping a reference
            self._fsdp_param_group = None

def _assert_all_rsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, RSDPModule):
            raise ValueError(f"Expects RSDPModule but got {type(module)}: {module}")