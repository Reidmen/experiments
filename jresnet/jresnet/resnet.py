from functools import partial
from typing import Any, Callable, Sequence, Type
from flax import linen as nn
from jax import Array
from jax._src.typing import ArrayLike
import jax.numpy as jnp
from jresnet.modules import ConvolutionBlock

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
    def __call__(self, x: Array) -> Array:
        partial_convolution = partial(
            self.convolution_class,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
        )
        first_width = (
            (8 * (x.shape[-1] + 1)) if self.adaptive_first_width else self.base_width
        )
        x = partial_convolution(first_width, strides=(2, 2))(x)
        x = partial_convolution(self.base_width, strides=(1, 1))(x)
        x = partial_convolution(self.base_width * 2, strides=(1, 1))(x)

        return x


def identity_array_operator(x: Array) -> Array:
    return x


class ResNetSkipConnection(nn.Module):
    strides: tuple[int, int]
    convolution_block_class: Type[ConvolutionBlock] = ConvolutionBlock
    identity: Callable[[Array], Array] = identity_array_operator

    @nn.compact
    def __call__(self, x: Array, output_shape: tuple[int, int]) -> Array:
        if x.shape != output_shape:
            x = self.convolution_block_class(
                output_shape[-1],
                kernel_size=(1, 1),
                strides=self.strides,
                activation=identity_array_operator,
            )(x)

        return x


class ResNetBlock(nn.Module):
    number_of_hidden: int
    strides: tuple[int, int] = (1, 1)
    activation: Callable[[ArrayLike], Array] = nn.relu
    convolution_block_class: Type[ConvolutionBlock] = ConvolutionBlock
    skip_class: Type[ResNetSkipConnection] = ResNetSkipConnection

    @nn.compact
    def __call__(self, x: Array) -> Array:
        skip_class = partial(
            self.skip_class,
            convolution_block_class=self.convolution_block_class,
        )
        y = self.convolution_block_class(
            self.number_of_hidden,
            padding=((1, 1), (1, 1)),
            strides=self.strides,
        )(x)
        y = self.convolution_block_class(
            self.number_of_hidden, padding=((1, 1), (1, 1)), is_last=True
        )(y)
        return self.activation(y + skip_class(self.strides)(x, y.shape))


class ResNetBottleneckBlock(nn.Module):
    number_of_hidden: int
    strides: tuple[int, int] = (1, 1)
    activation: Callable[[ArrayLike], Array] = nn.relu
    convolution_block_class: Type[ConvolutionBlock] = ConvolutionBlock
    skip_class: Type[nn.Module] = ResNetSkipConnection

    @nn.compact
    def __call__(self, x: Array) -> Array:
        y = self.convolution_block_class(
            self.number_of_hidden,
            padding=[(1, 1), (1, 1)],
            strides=self.strides,
        )(x)
        y = self.convolution_block_class(
            self.number_of_hidden, padding=[(1, 1), (1, 1)], is_last=True
        )(y)
        return self.activation(
            y
            + self.skip_class(
                self.strides,
                convolution_block_class=self.convolution_block_class,
            )(x, y.shape)
        )


def ResNetSequential(
    block_class: Type[ResNetBlock | ResNetBottleneckBlock],
    stage_sizes: Sequence[int],
    number_of_classes: int,
    hidden_sizes: Sequence[int] = (64, 128, 256, 512),
    convolution_block_class: Type[ConvolutionBlock] = ConvolutionBlock,
    base_class: Type[ResNetBase] = ResNetBase,
    norm_class: Type[nn.BatchNorm] | None = nn.BatchNorm,
) -> nn.Sequential:
    resnet_base = partial(base_class, convolution_block_class=convolution_block_class)
    layers: list[Callable[..., Array] | nn.Module] = [resnet_base()]

    for i, (hsize, number_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
        for block_element in range(number_blocks):
            strides: tuple[int, int] = (
                (1, 1) if i == 0 or block_element != 0 else (2, 2)
            )
            block_to_add = block_class(
                number_of_hidden=hsize,
                strides=strides,
                convolution_block_class=convolution_block_class,
            )
            layers.append(block_to_add)

    layers.append(partial(jnp.mean, axis=(1, 2)))
    layers.append(nn.Dense(number_of_classes))

    return nn.Sequential(layers)


def ResNet18(number_of_classes: int) -> nn.Sequential:
    return ResNetSequential(
        ResNetBlock,
        stage_sizes=STAGE_SIZES[18],
        number_of_classes=number_of_classes,
        base_class=ResNetBase,
    )


def ResNet34(number_of_classes: int) -> nn.Sequential:
    return ResNetSequential(
        ResNetBlock,
        stage_sizes=STAGE_SIZES[34],
        number_of_classes=number_of_classes,
        base_class=ResNetBase,
    )


def ResNet50(number_of_classes: int) -> nn.Sequential:
    return ResNetSequential(
        ResNetBlock,
        stage_sizes=STAGE_SIZES[50],
        number_of_classes=number_of_classes,
    )


def ResNet101(number_of_classes: int) -> nn.Sequential:
    return ResNetSequential(
        ResNetBottleneckBlock,
        stage_sizes=STAGE_SIZES[101],
        number_of_classes=number_of_classes,
        base_class=ResNetBase,
    )


def ResNet152(number_of_classes: int) -> nn.Sequential:
    return ResNetSequential(
        ResNetBottleneckBlock,
        stage_sizes=STAGE_SIZES[152],
        number_of_classes=number_of_classes,
        base_class=ResNetBase,
    )


def ResNet200(number_of_classes: int) -> nn.Sequential:
    return ResNetSequential(
        ResNetBottleneckBlock,
        stage_sizes=STAGE_SIZES[200],
        number_of_classes=number_of_classes,
        base_class=ResNetBase,
    )
