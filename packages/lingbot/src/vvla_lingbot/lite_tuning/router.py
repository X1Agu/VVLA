"""Frozen FP32 router weight, retaining the source parameter for invalidation."""
import torch
from torch import nn
from .state_guard import tensor_token


class CachedRouterLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        if original.bias is not None:
            raise ValueError('This recipe requires a bias-free router')
        self.source_weight = original.weight
        self.register_buffer('weight', original.weight.detach().float().clone())
        self.in_features, self.out_features = original.in_features, original.out_features
        self._source = (id(self.source_weight), tensor_token(self.source_weight))
        self._cached = (id(self.weight), tensor_token(self.weight))
        self.eval()

    def assert_fresh(self):
        if (id(self.source_weight), tensor_token(self.source_weight)) != self._source:
            raise RuntimeError('Router source changed; rebuild FP32 cache and engine')
        if (id(self.weight), tensor_token(self.weight)) != self._cached:
            raise RuntimeError('Router FP32 cache changed; rebuild engine')

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)
