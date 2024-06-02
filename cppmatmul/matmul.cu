#include <cuda_runtime.h>
#include <ostream>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>

template <typename T>
__global__ void vec_to_scalar(T *vec, T scalar, int num_elements) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    vec[idx] = vec[idx] * scalar;
  }
}

template <typename T>
void exec_vec_to_scalar(T *vec, T scalar, int num_elements) {
  dim3 dimBlock(256, 1, 1);
  dim3 dimGrid(ceil((T)num_elements) / dimBlock.x);

  vec_to_scalar<T><<<dimGrid, dimBlock>>>(vec, scalar, num_elements);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream str_stream;
    str_stream << "run_kernel lauch failed " << std::endl;
    str_stream << "dimBlock " << dimBlock.x << ", " << dimBlock.y << std::endl;
    str_stream << "dimGrid " << dimGrid.x << ", " << dimGrid.y << std::endl;
    str_stream << cudaGetErrorString(error);
    throw str_stream.str();
  }
}

template <typename T>
void map_vec_to_scalar(pybind11::array_t<T> vec, T scalar) {
  pybind11::buffer_info info = vec.request();

  if (info.ndim != 1) {
    std::stringstream stream;
    stream << "info.ndim != 1" << std::endl;
    stream << "infi.ndim: " << info.ndim << std::endl;
    throw std::runtime_error(stream.str());
  }

  int size = info.shape[0];
  int size_bytes = size * sizeof(T);
  T *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  T *ptr = reinterpret_cast<T *>(info.ptr);
  error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyDeviceToHost);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  exec_vec_to_scalar<T>(gpu_ptr, scalar, size);

  error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

PYBIND11_MODULE(matmul_cuda, matmul) {
  matmul.def("vec_to_scalar", map_vec_to_scalar<double>);
}
