from functools import partial
from typing import Any, Callable, Type
from flax import linen as nn
from jax import Array
from jresnet.modules import ConvolutionBlock, ConvolutionObject

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}


class ResNetBase(nn.Module):
    convolution_block_class: Type[nn.Module] = ConvolutionBlock

    @nn.compact
    def __call__(self, x: Array) -> Callable[..., Array]:
        return self.convolution_block_class(
            64, kernel_size=(7, 7), strides=(2, 2), padding=((3, 3), (3, 3))
        )(x)


class ResNetDeepBase(nn.Module):
    convolution_class: Type[nn.Module] = ConvolutionBlock
    base_width: int = 32
    adaptive_first_width: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Callable[..., Array]:
        partial_convolution = partial(
            self.convolution_class,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
        )
        first_width = (
            (8 * (x.shape[-1] + 1))
            if self.adaptive_first_width
            else self.base_width
        )
        x = partial_convolution(first_width, strides=(2, 2))(x)
        x = partial_convolution(self.base_width, strides=(1, 1))(x)
        x = partial_convolution(self.base_width * 2, strides=(1, 1))(x)

        return x
