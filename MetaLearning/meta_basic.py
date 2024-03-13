#!/usr/bin/env python3

import copy
import torch
import argparse
import dataclasses
import warnings

"""
High-level Wrappers
maml = l2l.algorithms.MAML(model, lr=0.1)
opt = torch.optim.SGD(maml.parameters(), lr=0.001)
for iteration in range(10):
    opt.zero_grad()
    task_model = maml.clone()  # torch.clone() for nn.Modules
    adaptation_loss = compute_loss(task_model)
    task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
    evaluation_loss = compute_loss(task_model)
    evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
    opt.step()

Low-Level Utilities

model = MyModel()
transform = l2l.optim.KroneckerTransform(l2l.nn.KroneckerLinear)
learned_update = l2l.optim.ParameterUpdate(  # learnable update function
        model.parameters(), transform)
clone = l2l.clone_module(model)  # torch.clone() for nn.Modules
error = loss(clone(X), y)
updates = learned_update(  # similar API as torch.autograd.grad
    error,
    clone.parameters(),
    create_graph=True,
)
l2l.update_module(clone, updates=updates)
loss(clone(X), y).backward()  # Gradients w.r.t model.parameters() and learned_update.parameters()

"""

def magic_box(x):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    The magic box operator, which evaluates to 1 but whose gradient is \\(dx\\):

    $$\\boxdot (x) = \\exp(x - \\bot(x))$$

    where \\(\\bot\\) is the stop-gradient (or detach) operator.

    This operator is useful when computing higher-order derivatives of stochastic graphs.
    For more informations, please refer to the DiCE paper. (Reference 1)

    **References**

    1. Foerster et al. 2018. "DiCE: The Infinitely Differentiable Monte-Carlo Estimator." arXiv.

    **Arguments**

    * **x** (Variable) - Variable to transform.

    **Return**

    * (Variable) - Tensor of 1, but it's gradient is the gradient of x.

    **Example**

    ~~~python
    loss = (magic_box(cum_log_probs) * advantages).mean()  # loss is the mean advantage
    loss.backward()
    ~~~
    """
    if isinstance(x, torch.Tensor):
        return torch.exp(x - x.detach())
    return x


def clone_parameters(param_list):
    return [p.clone() for p in param_list]


def clone_named_parameters(param_dict):
    return {k: p.clone() for k, p in param_dict.items()}

# clone_module`函数相比于使用`copy.deepcopy`
# torch.clone()：在PyTorch中，torch.clone()用于创建一个张量的深拷贝，这意味着被复制张量的数据和元数据（如形状和数据类型）都会被复制，
# 但得到的副本是一个全新的张量，与原始张量在内存中是分开的。使用torch.clone()是为了确保在对张量进行操作时不会影响到原始张量的数据。
# torch.clone()创建张量的深拷贝时，并不会复制原始张量的计算图和梯度历史
# copy.deepcopy()函数是Python标准库copy模块提供的方法，它能够创建对象的深拷贝，包括对象内部嵌套的所有对象。
# 对于PyTorch张量，copy.deepcopy()也能创建一个完全独立的张量副本。不过，这个方法更通用，可以用于Python中的任何对象，包括列表、字典、自定义对象等。
def clone_module(module, memo=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    克隆的模块会保持原始模块的计算图，允许你对新模块的参数相对于原始参数计算导数。??? 似乎不会保存计算图 该描述是错误的
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}
    # 保持计算图：由于使用了torch.clone()，新克隆的模块参数将保持与原始模块计算图的连接，这意味着可以对克隆后模块的参数进行反向传播，并影响到原始模块的参数
    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    # 1. 创建模块副本：
    # 首先，检查传入的对象是否是一个PyTorch模块。如果不是，直接返回该对象。
    # 使用Python的__new__方法创建模块的新实例，然后复制原始模块的__dict__、_parameters、_buffers和_modules属性。
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    # 2.处理参数：
    # 遍历所有参数，并检查它们是否已存在于memo映射中。如果是，直接使用映射中的克隆张量；如果不是，克隆参数并更新memo映射
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned
    # 3.处理缓冲区：
    # 类似于参数处理，但额外检查缓冲区是否需要梯度（requires_grad）。这对于诸如批归一化（BatchNorm）层的可学习参数很重要。
    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned
    # 4.递归克隆子模块：
    # 对于每个子模块，递归调用clone_module函数进行克隆，并更新_modules字典。
    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )
    # 5.重建扁平参数（主要用于RNNs）：==》特例分析
    # 对于使用扁平参数的模块（如RNNs），通过调用_apply方法来重建它们。这一步确保RNN及其变体能正确地克隆并保持其性能。
    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone

# 其目的是将一个PyTorch模块（及其所有子模块）的参数和缓冲区从计算图中分离（detach）。
# 这样做的结果是，对这些参数的任何操作都不会影响原始计算图中的梯度累积。这个函数主要用于处理克隆过的模块，使其可以独立于原始模块进行操作，而不影响原始模块的梯度。
def detach_module(module, keep_requires_grad=False):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Detaches all parameters/buffers of a previously cloned module from its computational graph.

    Note: detach works in-place, so it does not return a copy.

    **Arguments**

    * **module** (Module) - Module to be detached.
    * **keep_requires_grad** (bool) - By default, all parameters of the detached module will have
    `requires_grad` set to `False`. If this flag is set to `True`, then the `requires_grad` field
    will be the same as the pre-detached module.

    **Example**

    ~~~python
    net = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    detach_module(clone, keep_requires_grad=True)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate on clone, not net.
    ~~~
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            requires_grad = module._parameters[param_key].requires_grad
            detached = module._parameters[param_key].detach_()
            if keep_requires_grad and requires_grad:
                module._parameters[param_key].requires_grad_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()
            if keep_requires_grad:  # requires_grad checked above
                module._buffers[buffer_key].requires_grad_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key], keep_requires_grad=keep_requires_grad)


def clone_distribution(dist):
    # TODO: This function was never tested.
    clone = copy.deepcopy(dist)

    for param_key in clone.__dict__:
        item = clone.__dict__[param_key]
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                clone.__dict__[param_key] = dist.__dict__[param_key].clone()
        elif isinstance(item, torch.nn.Module):
            clone.__dict__[param_key] = clone_module(dist.__dict__[param_key])
        elif isinstance(item, torch.distributions.Distribution):
            clone.__dict__[param_key] = clone_distribution(dist.__dict__[param_key])

    return clone


def detach_distribution(dist):
    # TODO: This function was never tested.
    for param_key in dist.__dict__:
        item = dist.__dict__[param_key]
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                dist.__dict__[param_key] = dist.__dict__[param_key].detach()
        elif isinstance(item, torch.nn.Module):
            dist.__dict__[param_key] = detach_module(dist.__dict__[param_key])
        elif isinstance(item, torch.distributions.Distribution):
            dist.__dict__[param_key] = detach_distribution(dist.__dict__[param_key])
    return dist


def update_module(module, updates=None, memo=None):
    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Updates the parameters of a module in-place, in a way that preserves differentiability.

    The parameters of the module are swapped with their update values, according to:
    \[
    p \gets p + u,
    \]
    where \(p\) is the parameter, and \(u\) is its corresponding update.


    **Arguments**

    * **module** (Module) - The module to update.
    * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the tensors in .update attributes.

    **Example**
    ~~~python
    error = loss(model(X), y)
    grads = torch.autograd.grad(
        error,
        model.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(model, updates=updates)
    ~~~
    """
    # 实际更新模型参数。这个函数遍历模型的所有参数和缓冲区，如果它们有.update属性，则应用这些更新。
    # memo: 用于避免在递归更新中重复更新同一个参数的字典 是否后期可以运用到qkv的共享上？
    if memo is None:
        memo = {}
    # 适用于有update的情况 获取参数的update属性
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            # 防止参数重新更新
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule ==》实际分析是否有？
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module


def accuracy(preds, targets):
    """Computes accuracy"""
    acc = (preds.argmax(dim=1).long() == targets.long()).sum().float()
    return acc / preds.size(0)


def flatten_config(args, prefix=None):
    flat_args = dict()
    if isinstance(args, argparse.Namespace):
        args = vars(args)
        return flatten_config(args)
    elif not dataclasses.is_dataclass(args) and not isinstance(args, dict):
        flat_args[prefix] = args
        return flat_args
    elif dataclasses.is_dataclass(args):
        keys = dataclasses.fields(args)
        def getvalue(x): return getattr(args, x.name)
    elif isinstance(args, dict):
        keys = args.keys()
        def getvalue(x): return args[x]
    else:
        raise 'Unknown args'
    for key in keys:
        value = getvalue(key)
        if prefix is None:
            if isinstance(key, str):
                prefix_child = key
            elif isinstance(key, dataclasses.Field):
                prefix_child = key.name
            else:
                raise 'Unknown key'
        else:
            prefix_child = prefix + '.' + key.name
        flat_child = flatten_config(value, prefix=prefix_child)
        flat_args.update(flat_child)
    return flat_args


class _ImportRaiser(object):

    def __init__(self, name, command):
        self.name = name
        self.command = command

    def raise_import(self):
        msg = self.name + ' required. Try: ' + self.command
        raise ImportError(msg)

    def __getattr__(self, *args, **kwargs):
        self.raise_import()

    def __call__(self, *args, **kwargs):
        self.raise_import()


class _SingleWarning(object):

    def __init__(self):
        self.warned_messages = []
        self.warning_categories = {
            'default': UserWarning,
            'deprecation': DeprecationWarning,
        }

    def __call__(self, message, severity=None):
        if message not in self.warned_messages:
            if severity is None:
                severity = 'default'
            if severity == 'error':
                raise RuntimeError(message)
            elif isinstance(severity, str):
                severity = self.warning_categories[severity]
            warnings.warn(message, category=severity)
            self.warned_messages.append(message)


warn_once = _SingleWarning()
