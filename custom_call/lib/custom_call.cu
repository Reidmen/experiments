#include "helpers.hpp"
#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

#ifdef __CUDACC__
#define CUSTOM_JAX __host__ __device__
#else
#define CUSTOM_JAX inline
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace custom_call_jax {

struct CustomDescriptor {
  int64_t size;
};

template <typename T> inline void sin_cos(const T &x, T *sx, T *cx) {
  *sx = sin(x);
  *cx = cos(x);
}

template <typename T>
__host__ __device__ void custom_computation(const T &mean, const T &ecc,
                                            T *sin_ecc, T *cos_ecc) {
  const T tol = 1e-12;
  T g, E = (mean < M_PI) ? mean + 0.85 + ecc : mean - 0.85 * ecc;
  for (int i = 0; i < 20; i++) {
    sincos(E, sin_ecc, sin_cos);
    g = E - ecc * (*sin_ecc) - mean;
    if (fabs(g) <= tol)
      return;
    E -= g / (1 - ecc * (*cos_ecc));
  }
}

template <typename T>
__global__ void custom_kernel(int64_t size, const T *mean, const T *ecc,
                              T *sin_ecc, T *cos_ecc) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    custom_computation<T>(mean[idx], ecc[idx], sin_ecc + idx, cos_ecc + idx);
  }
}

void throw_error(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void apply_custom_call(cudaStream_t stream, void **buffers,
                              const char *opaque, size_t opaque_length) {}

void custom_call_f32(cudaStream_t stream, void **buffers, const char *opaque,
                     size_t opaque_length) {
  apply_custom_call<float>(stream, buffers, opaque, opaque_length);
}
void custom_call_f64(cudaStream_t stream, void **buffers, const char *opaque,
                     size_t opaque_length) {
  apply_custom_call<double>(stream, buffers, opaque, opaque_length);
}

// pybind11 wrappers to expose in the Python interpreter
template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptor(descriptor));
}

template <typename T> pybind11::capsule EncapsulateFunction(T *function) {
  return pybind11::capsule(custom_helpers::bit_cast<void *>(function),
                           "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["custom_call_f32"] = EncapsulateFunction(custom_call_f32);
  dict["custom_call_f64"] = EncapsulateFunction(custom_call_f64);
  return dict;
}

PYBIND11_MODULE(custom_call, module) {
  module.def("registrations", &Registrations);
  module.def("build_descriptor", [](std::int64_t size) {
    return PackDescriptor(CustomDescriptor{size});
  });
}

} // namespace custom_call_jax
