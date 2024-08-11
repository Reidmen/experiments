from functools import partial
from jax import Array, dtypes, numpy as jnp
from jax._src import core
from jax._src.core import ShapedArray
from jax._src.typing import ArrayLike
from jax.lib import xla_client
from jax.interpreters import mlir, xla
from jaxlib.mlir.ir import OpResult
import numpy

try:
    from . import gpu_custom_operation
except ImportError:
    gpu_custom_operation = None

# register public operations
for _name, value_ in gpu_custom_operation.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


# requied for jit compilation
def _custom_operation_abstract(
    mean_: Array, ecc_: Array
) -> tuple[ShapedArray, ShapedArray]:
    shape = mean_.shape
    dtype = dtypes.canonicalize_dtype(mean_.dtype)
    assert dtypes.canonicalize_dtype(ecc_.dtype) == dtype
    assert ecc_.shape == shape
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))


# required lowering for MLIR -> expose the primitive to the jax XLA backend
def _custom_operation_lowering(ctx, mean, ecc, *, platform="cpu") -> OpResult:
    assert mean.type == ecc.type
    mean = ctx.avals_in
    nptype = numpy.dtype(mean.dtype)
    irtype = mlir.ir.RankedTensorType(mean.type)
    dims = irtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))
    size = numpy.prod(dims).astype(numpy.int64)

    if nptype == numpy.float32:
        operation_name = platform + "_custom_f32"
    elif nptype == numpy.float64:
        operation_name = platform + "_custom_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {nptype}")

    if platform == "gpu":
        if gpu_custom_operation is None:
            raise ValueError("custom_call module not compiled with CUDA")
        opaque = gpu_custom_operation.build_descriptor(size)

        return mlir.custom_call(
            operation_name,
            result_types=[irtype, irtype],
            operands=[mean, ecc],
            operand_layouts=[layout, layout],
            result_layouts=[layout, layout],
            backend_config=opaque,
        ).result
    else:
        raise ValueError("Unsupported platform: only 'gpu'")


# register primitive
_custom_primitive = core.Primitive("custom_call")
_custom_primitive.multiple_results = True
_custom_primitive.def_impl(partial(xla.apply_primitive, _custom_primitive))
_custom_primitive.def_abstract_eval(_custom_operation_abstract)

# connect XLA translation rules for JIT compilation
mlir.register_lowering(
    _custom_primitive,
    partial(_custom_operation_lowering, platform="gpu"),
    platform="gpu",
)
# TODO Add jvp and batching rules
