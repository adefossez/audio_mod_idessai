# MIT License, copyright 2024 Alexandre Défossez.
"""Utilities for Pytorch, in particular:

- A robust activation checkpointing, that should work in any situation (torch.compile, FSDP, etc.).
    It just doesn't handle RNG state, so do not use it with dropouts etc.
- A mostly untested LoRA implementation based on forward hooks.

In particular the checkpointing works nicely with LoRA, while the one from PyTorch would complain.
"""
import inspect
import math
import typing as tp

import torch
from torch import nn


class Checkpoint(torch.autograd.Function):
    """See `simple_checkpoint`."""
    @staticmethod
    def forward(ctx, function, *args) -> tp.Any:
        to_save = []
        ctx.others = []
        ctx.function = function
        # Sources will indicate whether the arg in position N is
        # a tensor stored in ctx.save_for_backward, or inside ctx.others.
        ctx.sources = []
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                to_save.append(arg)
                ctx.sources.append('tensor')
                new_args.append(arg.detach())
            else:
                ctx.sources.append('other')
                ctx.others.append(arg)
                new_args.append(arg)
        ctx.save_for_backward(*to_save)
        # During the forward, we just make a pass with no gradient computed.
        with torch.no_grad():
            res = function(*new_args)
        return res

    @staticmethod
    def backward(ctx, *grads) -> tp.Tuple[tp.Optional[torch.Tensor], ...]:
        pseudo_tensors = []
        with torch.set_grad_enabled(True):
            # We create leaf tensors to collect the output gradients.
            # We call them pseudo_tensors because they are pretending to be the input
            # to `function` but are not directly
            for tensor in ctx.saved_tensors:
                pseudo_tensor = tensor.detach()
                pseudo_tensor.requires_grad_(True)
                pseudo_tensors.append(pseudo_tensor)
            pseudo_tensors_copy = list(pseudo_tensors)
            args = []
            for source in ctx.sources:
                if source == 'other':
                    args.append(ctx.others.pop(0))
                else:
                    assert source == 'tensor'
                    args.append(pseudo_tensors_copy.pop(0))
            res = ctx.function(*args)
            # The second forward with grad computation allows us to connect the input leaf tensors
            # inside pseudo_tensors, to the outputs of the function called.
        if not isinstance(res, tuple):
            res = (res,)
        # Now we just ask Torch to compute the derivative of `res` given the gradient coming from above
        # `grads`. The computed gradient will end up into the `pseudo_tensors` grad attributes.
        torch.autograd.backward(res, grads)
        out: tp.List[tp.Optional[torch.Tensor]] = [None]
        for source in ctx.sources:
            # We still need to output `None` values for non tensor parameters.
            if source == 'other':
                out.append(None)
            else:
                assert source == 'tensor'
                out.append(pseudo_tensors.pop(0).grad)
        return tuple(out)


def simple_checkpoint(module: torch.nn.Module, *args, **kwargs):
    """Custom implementation of checkpointing in PyTorch as the builtin implementation is broken
    when using torch compile. Only supports wrapping a `nn.Module` with a forward with no `*args` or `**kwargs`.

    https://github.com/pytorch/pytorch/issues/97436.
    Should be resolved in nightlies, but it is quite fun and simple to code it ourselves.
    """
    if hasattr(module, '_fsdp_wrapped_module'):
        module_for_sig = module._fsdp_wrapped_module
    else:
        module_for_sig = module
    sig = inspect.signature(module_for_sig.forward)
    # We first flatten all arguments to use only *args, to make things easier and because
    # torch.autograd.Function has weird support for kwargs.
    bounded = sig.bind(*args, **kwargs)
    new_args = []
    for name, param in sig.parameters.items():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            raise RuntimeError("simple_checkpoint doesn't support var args.")
        if name in bounded.arguments:
            new_args.append(bounded.arguments[name])
        elif param.default is not inspect.Parameter.empty:
            new_args.append(param.default)
        else:
            raise RuntimeError(f"Missing value for {name}.")

    return Checkpoint.apply(module, *new_args)


def add_lora_(module: nn.Module, r:int = 8, alpha: float = 1.0, min_numel: float = 1e6):
    """Simple implementation for LoRA (https://arxiv.org/pdf/2106.09685).
    It adds a hook that will populate all the 2D parameter tensors (of at least min_numel size)
    with a LoRA extended version when entering the forward phase, and will remove it at the end of the forward.

    Note that the hook is directly set on `module` which will populate ALL params. To avoid blowing up
    your memory, you might want to call this independently for different layers.

    Args:
        module (nn.Module): module to LoRAify.
        r (int): low rank dimension.
        alpha (float): scaling to the `a @ b` additive term.
        min_numel (float): any tensor with less than than many params will be ignored.

    ..Note:: Only tensors with a required_grad=True will be considered for LoRAification.

    ..Note:: I chose to implement LoRA by explicitely populating the weights rather than trying
        to correct the output of the original linear layers. In terms of complexity, if we have
        N the number of items (e.g. Batch * Nb time steps for a transformer), D the dimension,
        and R the low rank dimension, then the complexity of changing the weights is:
            D^2 r + N D^2
        vs. for correcting the outputs:
            2 N D r + N D^2
        So although in general 2 N D r < D^2 r, it is not by much, and in any case N D^2 dominates
        for any training case. For inference with N = 1, then one should just remove LoRA entirely.
        I find this way to be more generic, as it doesn't matter how the underlying module use the weights,
        in particular they might implement more complex logic than just having a Linear layer.
    """
    params: list[tuple[nn.Module, str]] = []

    def add_param(mod: nn.Module, name: str):
        # LoRAify a new param, adding the `a` and `b` tensors, and removing
        # the original tensor name.
        weight = mod._parameters.pop(name)
        weight.requires_grad_(False)
        C, D = weight.shape
        lora_b = torch.zeros(C, r, device=weight.device, dtype=weight.dtype)
        lora_a = torch.zeros(r, D, device=weight.device, dtype=weight.dtype)
        if isinstance(mod, nn.Embedding):
            # Then lora_b is [num_embs, r] and lora_a is [r, dim]
            nn.init.normal_(lora_b)
        else:
            # Then lora_b is [chout, r] and lora_a is [r, chin]
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        setattr(mod, name + "_lora_a", nn.Parameter(lora_a))
        setattr(mod, name + "_lora_b", nn.Parameter(lora_b))
        setattr(mod, name + "_lora_base", nn.Parameter(weight, requires_grad=False))
        params.append((mod, name))

    def pre_hook(_, args) -> None:
        # Populate the original tensor name with `weight + a @ b` for all
        # the LoRAified weights.
        for mod, name in params:
            if name in mod._parameters:
                return
            a = getattr(mod, name + "_lora_a")
            b = getattr(mod, name + "_lora_b")
            base = getattr(mod, name + "_lora_base")
            # if weight is [D, F]
            # a is [r, F]
            # b is [D, r]
            mod._parameters[name] = torch.addmm(base, b, a, alpha=alpha / r)

    def post_hook(_, args, output) -> None:
        # Delete all the intermediate weight tensors.
        for mod, name in params:
            mod._parameters.pop(name, None)

    for mod in module.modules():
        for name, param in list(mod.named_parameters(recurse=False)):
            if '_lora_' in name:
                continue
            if param.numel() >= min_numel and param.requires_grad and param.dim() == 2:
                add_param(mod, name)

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)
    return module
