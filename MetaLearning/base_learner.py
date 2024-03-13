#!/usr/bin/env python3

from torch import nn


class BaseLearner(nn.Module):
    # BaseLearner类提供了一种灵活的方式来包装任何PyTorch模块，使其可以作为一个基础学习器使用。通过重写__getattr__方法，它能够无缝地代理对内部模块的属性和方法的访问。
    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.module = module

    def __getattr__(self, attr):
        try:
            return super(BaseLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
