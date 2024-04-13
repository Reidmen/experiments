from typing import Sequence
import jax
from jax._src.tree_util import Leaf
import jax.numpy as jnp
import pytest
from flax import linen as nn
from jresnet.resnet import ResNet101, ResNet18, ResNet34, ResNet50

# github.com/google/flax/blob/master/linen_examples/imagenet/models.py
# matches https://pytorch.org/hub/pytorch_vision_resnet/
RESNET_PARAM_COUNTS = {
    18: 11689512,
    34: 21797672,
    50: 25557032,
    101: 44549160,
    152: 60192808,
}
# https://pytorch.org/hub/pytorch_vision_wide_resnet/
WIDE_RESNET_PARAM_COUNTS = {50: 68883240, 101: 126886696}

# https://pytorch.org/hub/pytorch_vision_resnext/
RESNEXT_PARAM_COUNTS = {50: 25028904, 101: 88791336}

# https://docs.fast.ai/vision.models.xresnet.html
RESNETD_PARAM_COUNTS = {
    18: 11708744,
    34: 21816904,
    50: 25576264,
    101: 44568392,
    152: 60212040,
}

# github.com/zhanghang1989/ResNeSt (torch version)
RESNEST_PARAM_COUNTS = {
    50: 27483240,
    101: 48275016,
    200: 70201544,
    269: 110929480,
}


def number_of_parameters(
    model: nn.Module, init_shape: Sequence[int] = (1, 224, 224, 3)
) -> Leaf:
    init_array = jnp.ones(init_shape, jnp.float32)
    params = model.init(jax.random.PRNGKey(0), init_array)["params"]
    return sum(map(jnp.size, jax.tree_util.tree_leaves(params)))


def _test_resnet_parameter_count(size: int) -> None:
    model = eval(f"ResNet{size}")(number_of_classes=1000)
    assert number_of_parameters(model) == RESNET_PARAM_COUNTS[size]


@pytest.mark.parametrize("size", [18, 50])
def test_resnet_parameter_count(size: int) -> None:
    _test_resnet_parameter_count(size)


@pytest.mark.slow
@pytest.mark.parametrize("size", [34, 101, 152])
def test_resnet_parameter_count_slow(size: int) -> None:
    _test_resnet_parameter_count(size)


@pytest.mark.parametrize(
    "start, end", [(0, 5), (0, None), (0, -3), (4, -2), (3, -1), (2, None)]
)
def test_slice_variables(start: int, end: int | None) -> None:
    model = ResNet18(number_of_classes=10)
    key = jax.random.PRNGKey(0)
    variables = model.init(key, jnp.ones((1, 224, 224, 3)))
    sliced_variables = slice_variables(variables, start, end)
    sliced_model = nn.Sequential(model.layers[start:end])
    first = variables["params"][f"layers_{start}"]["ConvBlock_0"]["Conv_0"][
        "kernel"
    ]
    assert isinstance(first, jax.Array)
    slice_inp = jnp.ones((1, 224, 224, first.shape[2]))
    exp_sliced_vars = sliced_model.init(key, slice_inp)

    assert set(sliced_variables["params"].keys()) == set(
        exp_sliced_vars["params"].keys()
    )
    assert set(sliced_variables["batch_stats"].keys()) == set(
        exp_sliced_vars["batch_stats"].keys()
    )

    assert jax.tree_map(jnp.shape, sliced_variables) == jax.tree_map(
        jnp.shape, exp_sliced_vars
    )

    sliced_model.apply(sliced_variables, slice_inp, mutable=False)
