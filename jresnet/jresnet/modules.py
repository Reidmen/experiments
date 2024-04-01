from __future__ import annotations
from typing import Any, Callable, Iterable, TypeAlias

import flax.linen as nn
from jax import Array
from jax.nn.initializers import Initializer

InitObject: TypeAlias = Callable[..., Initializer]
ConvolutionObject: TypeAlias = Callable[..., nn.Conv]


class ConvolutionBlock(nn.Module):
    numer_filters: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    activation: Callable[..., Array] = nn.relu
    padding: str | Iterable[tuple[int, int]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    kernel_initialized: InitObject = nn.initializers.kaiming_normal
    bias_initialized: InitObject = nn.initializers.zeros

    convolution_class: ConvolutionObject = nn.Conv
    force_convolution_bias: bool = False

    @nn.compact
    def __call__(self, x: Array, *args, **kwargs) -> Any:
        super().__call__(*args, **kwargs)
        x = self.convolution_class(
            self.numer_filters,
            self.kernel_size,
            self.strides,
            use_bias=(
                not self.convolution_class or self.force_convolution_bias
            ),
            padding=self.padding,
            feature_group_count=self.groups,
            kernel_init=self.kernel_initialized,
            bias_init=self.bias_initialized,
        )(x)
        if not self.is_last:
            x = self.activation(x)

        return x
