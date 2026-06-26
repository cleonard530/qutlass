#pragma once

#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/util/Exception.h>

#include "cutlass/cutlass.h"

inline cudaStream_t get_current_cuda_stream(int device_index) {
  void* stream_ptr = nullptr;
  aoti_torch_get_current_cuda_stream(device_index, &stream_ptr);
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

#define CUTLASS_CHECK(status)                              \
  {                                                        \
    cutlass::Status error = status;                        \
    STD_TORCH_CHECK(error == cutlass::Status::kSuccess,    \
                    cutlassGetStatusString(error));         \
  }

#define CUDA_CHECK(status)                                           \
  {                                                                  \
    cudaError_t error = status;                                      \
    STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error)); \
  }

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

int32_t get_sm_version_num();

template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};
