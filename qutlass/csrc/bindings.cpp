/*
 * Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "include/gemm.h"
#include "include/fused_quantize_host.h"
#include "include/backward_host.h"

#include <torch/csrc/stable/library.h>

#include <vector>
#include <iostream>
#include <utility>
#include <initializer_list>

namespace QUTLASS {

namespace {

struct TensorArg {
  const torch::stable::Tensor& tensor;
  const char* name;
  int index;
};

void check_all_contiguous(const char* op, std::initializer_list<TensorArg> args) {
  for (const auto& arg : args) {
    STD_TORCH_CHECK(arg.tensor.is_contiguous(), op, ": expected ", arg.name, " to be contiguous");
  }
}

void check_device_type_cuda(const char* op, std::initializer_list<std::reference_wrapper<const torch::stable::Tensor>> tensors) {
  for (const auto& t : tensors) {
    STD_TORCH_CHECK(
        t.get().device().type() == torch::headeronly::DeviceType::CUDA,
        op, ": expected a CUDA tensor");
  }
}

void check_all_same_gpu(const char* op, std::initializer_list<TensorArg> args) {
  if (args.size() == 0) {
    return;
  }
  auto it = args.begin();
  const auto expected = it->tensor.get_device_index();
  for (const auto& arg : args) {
    STD_TORCH_CHECK(
        arg.tensor.get_device_index() == expected,
        op, ": expected ", arg.name, " to be on the same GPU");
  }
}

}  // namespace


torch::stable::Tensor matmul_mxf4_bf16_tn(torch::stable::Tensor const& A,
                                  torch::stable::Tensor const& B,
                                  torch::stable::Tensor const& A_sf,
                                  torch::stable::Tensor const& B_sf,
                                  torch::stable::Tensor const& alpha)
{
    check_all_contiguous("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    check_device_type_cuda("matmul_mxf4_bf16_tn", {A, B, A_sf, B_sf, alpha});
    check_all_same_gpu("matmul_mxf4_bf16_tn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3},
                                                   {alpha, "alpha", 4}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::Byte, "A must be uint8");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::Byte, "B must be uint8");
    STD_TORCH_CHECK(A_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "A_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(B_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "B_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    STD_TORCH_CHECK(A.size(1) >= 32, "A K-dim must be >= 32");
    STD_TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto OUT   = torch::stable::new_empty(A, {M, N}, torch::headeronly::ScalarType::BFloat16);

    matmul_host_mxf4_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

torch::stable::Tensor matmul_nvf4_bf16_tn(torch::stable::Tensor const& A,
                                  torch::stable::Tensor const& B,
                                  torch::stable::Tensor const& A_sf,
                                  torch::stable::Tensor const& B_sf,
                                  torch::stable::Tensor const& alpha)
{
    check_all_contiguous("matmul_nvf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    check_device_type_cuda("matmul_nvf4_bf16_tn", {A, B, A_sf, B_sf, alpha});
    check_all_same_gpu("matmul_nvf4_bf16_tn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3},
                                                   {alpha, "alpha", 4}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::Byte, "A must be uint8");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::Byte, "B must be uint8");
    STD_TORCH_CHECK(A_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "A_sf must be float8_e4m3fn");
    STD_TORCH_CHECK(B_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "B_sf must be float8_e4m3fn");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    STD_TORCH_CHECK(A.size(1) >= 16, "A K-dim must be >= 16");
    STD_TORCH_CHECK(B.size(1) >= 16, "B K-dim must be >= 16");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto OUT   = torch::stable::new_empty(A, {M, N}, torch::headeronly::ScalarType::BFloat16);

    matmul_host_nvf4_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

torch::stable::Tensor matmul_ada_mxf4_bf16_tn(torch::stable::Tensor const&A,
                                      torch::stable::Tensor const&B,
                                      torch::stable::Tensor const&A_sf,
                                      torch::stable::Tensor const&B_sf,
                                      torch::stable::Tensor const& alpha)
{
    check_all_contiguous("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3}});
    check_device_type_cuda("matmul_ada_mxf4_bf16_tn", {A, B, A_sf, B_sf, alpha});
    check_all_same_gpu("matmul_ada_mxf4_bf16_tn", {{A, "A", 0},
                                                       {B, "B", 1},
                                                       {A_sf, "A_sf", 2},
                                                       {B_sf, "B_sf", 3},
                                                       {alpha, "alpha", 4}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::Byte, "A must be uint8");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::Byte, "B must be uint8");
    STD_TORCH_CHECK(A_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "A_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(B_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "B_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    STD_TORCH_CHECK(A.size(1) >= 32, "A K-dim must be >= 32");
    STD_TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::stable::new_empty(A, {M, N}, torch::headeronly::ScalarType::BFloat16);

    matmul_host_ada_mxf4_bf16_tn(A, B, A_sf, B_sf, C, alpha);

    return C;
}

torch::stable::Tensor matmul_mxf8_bf16_tn(torch::stable::Tensor const& A,
                                  torch::stable::Tensor const& B,
                                  torch::stable::Tensor const& A_sf,
                                  torch::stable::Tensor const& B_sf,
                                  torch::stable::Tensor const& alpha)
{
    check_all_contiguous("matmul_mxf8_bf16_tn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3},
                                                      {alpha, "alpha", 4}});
    check_device_type_cuda("matmul_mxf8_bf16_tn", {A, B, A_sf, B_sf, alpha});

    check_all_same_gpu("matmul_mxf8_bf16_tn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3},
                                                   {alpha, "alpha", 4}});

    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "A must be float8_e4m3fn");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "B must be float8_e4m3fn");
    STD_TORCH_CHECK(A_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "A_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(B_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "B_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match for A @ B.T");
    STD_TORCH_CHECK(A.size(1) >= 32, "A K-dim must be >= 32");
    STD_TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto OUT = torch::stable::new_empty(A, {M, N}, torch::headeronly::ScalarType::BFloat16);

    matmul_host_mxf8_bf16_tn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

torch::stable::Tensor matmul_mxf8_bf16_nn(torch::stable::Tensor const& A,
                                  torch::stable::Tensor const& B,
                                  torch::stable::Tensor const& A_sf,
                                  torch::stable::Tensor const& B_sf,
                                  torch::stable::Tensor const& alpha)
{
    check_all_contiguous("matmul_mxf8_bf16_nn", {{A, "A", 0},
                                                      {B, "B", 1},
                                                      {A_sf, "A_sf", 2},
                                                      {B_sf, "B_sf", 3},
                                                      {alpha, "alpha", 4}});
    check_device_type_cuda("matmul_mxf8_bf16_nn", {A, B, A_sf, B_sf, alpha});

    check_all_same_gpu("matmul_mxf8_bf16_nn", {{A, "A", 0},
                                                   {B, "B", 1},
                                                   {A_sf, "A_sf", 2},
                                                   {B_sf, "B_sf", 3},
                                                   {alpha, "alpha", 4}});

    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "A must be float8_e4m3fn");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "B must be float8_e4m3fn");
    STD_TORCH_CHECK(A_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "A_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(B_sf.scalar_type() == torch::headeronly::ScalarType::Float8_e8m0fnu, "B_sf must be float8_e8m0fnu");
    STD_TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    STD_TORCH_CHECK(A.size(0) == B.size(1), "Inner dimensions must match for A.T @ B.T");
    STD_TORCH_CHECK(A.size(0) >= 32, "A K-dim must be >= 32");
    STD_TORCH_CHECK(B.size(1) >= 32, "B K-dim must be >= 32");

    uint32_t M = A.size(1);
    uint32_t N = B.size(0);
    auto OUT = torch::stable::new_empty(A, {M, N}, torch::headeronly::ScalarType::BFloat16);

    matmul_host_mxf8_bf16_nn(OUT, A, B, A_sf, B_sf, alpha);

    return OUT;
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fusedQuantizeMxQuest(torch::stable::Tensor const& A,
                                                              torch::stable::Tensor const& B,
                                                              torch::stable::Tensor& OUT,
                                                              torch::stable::Tensor& OUT_sf)
{
    check_all_contiguous("fusedQuantizeMxQuest", {{A, "A", 0},
                                                       {B, "B", 1},
                                                       {OUT, "OUT", 2},
                                                       {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeMxQuest", {A, B, OUT, OUT_sf});
    check_all_same_gpu("fusedQuantizeMxQuest", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {OUT, "OUT", 2},
                                                    {OUT_sf, "OUT_sf", 3}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==32){
        fusedQuantizeMxQuest_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==64){
        fusedQuantizeMxQuestHad64_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==128){
        fusedQuantizeMxQuestHad128_host(OUT, OUT_sf, A, B);
    } else {
        STD_TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> fusedQuantizeMxQuestWithMask(
                                                                    torch::stable::Tensor const& A,
                                                                    torch::stable::Tensor const& B,
                                                                    torch::stable::Tensor& OUT,
                                                                    torch::stable::Tensor& OUT_sf,
                                                                    torch::stable::Tensor& OUT_mask)
{
    check_all_contiguous("fusedQuantizeMxQuestWithMask", {{A, "A", 0},
                                                               {B, "B", 1},
                                                               {OUT, "OUT", 2},
                                                               {OUT_sf, "OUT_sf", 3},
                                                               {OUT_mask, "OUT_mask", 4}});
    check_device_type_cuda("fusedQuantizeMxQuestWithMask", {A, B, OUT, OUT_sf, OUT_mask});
    check_all_same_gpu("fusedQuantizeMxQuestWithMask", {{A, "A", 0},
                                                            {B, "B", 1},
                                                            {OUT, "OUT", 2},
                                                            {OUT_sf, "OUT_sf", 3},
                                                            {OUT_mask, "OUT_mask", 4}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==32){
        fusedQuantizeMxQuestWithMask_host(OUT, OUT_sf, OUT_mask, A, B);
    } else {
        STD_TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 32.");
    }

    return std::make_tuple(OUT, OUT_sf, OUT_mask);
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fusedQuantizeMxAbsMax(torch::stable::Tensor const& A,
                                                               torch::stable::Tensor const& B,
                                                               torch::stable::Tensor& OUT,
                                                               torch::stable::Tensor& OUT_sf)
{
    check_all_contiguous("fusedQuantizeMxAbsMax", {{A, "A", 0},
                                                        {B, "B", 1},
                                                        {OUT, "OUT", 2},
                                                        {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeMxAbsMax", {A, B, OUT, OUT_sf});
    check_all_same_gpu("fusedQuantizeMxAbsMax", {{A, "A", 0},
                                                     {B, "B", 1},
                                                     {OUT, "OUT", 2},
                                                     {OUT_sf, "OUT_sf", 3}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==32){
        fusedQuantizeMxAbsMax_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==64){
        fusedQuantizeMxAbsMaxHad64_host(OUT, OUT_sf, A, B);
    } else if(HAD_GS==128){
#if TARGET_CUDA_ARCH == 100
        auto global_scale = torch::stable::full({1}, 0.0, torch::headeronly::ScalarType::Float, std::nullopt, A.device()); //FIXME: add input global_scale to interface for consistency
        fusedQuantizeMxAbsMax_host_sm100(OUT, OUT_sf, A, B, global_scale);
#elif TARGET_CUDA_ARCH == 120
        fusedQuantizeMxAbsMaxHad128_host(OUT, OUT_sf, A, B);
#endif

    } else {
        STD_TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fusedQuantizeNvQuest(torch::stable::Tensor const& A,
                                                         torch::stable::Tensor const& B,
                                                         torch::stable::Tensor& OUT,
                                                         torch::stable::Tensor& OUT_sf,
                                                         torch::stable::Tensor const& global_scale)
{
    check_all_contiguous("fusedQuantizeNvQuest", {{A, "A", 0},
                                                  {B, "B", 1},
                                                  {OUT, "OUT", 2},
                                                  {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeNvQuest", {A, B, OUT, OUT_sf, global_scale});
    check_all_same_gpu("fusedQuantizeNvQuest", {{A, "A", 0},
                                               {B, "B", 1},
                                               {OUT, "OUT", 2},
                                               {OUT_sf, "OUT_sf", 3},
                                               {global_scale, "global_scale", 4}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(global_scale.scalar_type() == torch::headeronly::ScalarType::Float, "global_scale must be float");
    STD_TORCH_CHECK(global_scale.dim() == 1 && global_scale.size(0) == 1, "global_scale must be a scalar");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==16){
        fusedQuantizeNvQuest_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==32){
        fusedQuantizeNvQuestHad32_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==64){
        fusedQuantizeNvQuestHad64_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==128){
        fusedQuantizeNvQuestHad128_host(OUT, OUT_sf, A, B, global_scale);
    } else {
        STD_TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 16, 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> fusedQuantizeNvAbsMax(torch::stable::Tensor const& A,
                                                         torch::stable::Tensor const& B,
                                                         torch::stable::Tensor& OUT,
                                                         torch::stable::Tensor& OUT_sf,
                                                         torch::stable::Tensor const& global_scale)
{
    check_all_contiguous("fusedQuantizeNvAbsMax", {{A, "A", 0},
                                                  {B, "B", 1},
                                                  {OUT, "OUT", 2},
                                                  {OUT_sf, "OUT_sf", 3}});
    check_device_type_cuda("fusedQuantizeNvAbsMax", {A, B, OUT, OUT_sf, global_scale});
    check_all_same_gpu("fusedQuantizeNvAbsMax", {{A, "A", 0},
                                               {B, "B", 1},
                                               {OUT, "OUT", 2},
                                               {OUT_sf, "OUT_sf", 3},
                                               {global_scale, "global_scale", 4}});
    STD_TORCH_CHECK(A.scalar_type() == torch::headeronly::ScalarType::BFloat16, "A must be bf16");
    STD_TORCH_CHECK(B.scalar_type() == torch::headeronly::ScalarType::BFloat16, "B must be bf16");
    STD_TORCH_CHECK(global_scale.scalar_type() == torch::headeronly::ScalarType::Float, "global_scale must be float");
    STD_TORCH_CHECK(global_scale.dim() == 1 && global_scale.size(0) == 1, "global_scale must be a scalar");
    STD_TORCH_CHECK(B.size(0) == B.size(1), "Rotation matrix must be square");

    uint32_t HAD_GS = B.size(0);
    STD_TORCH_CHECK((A.numel()%HAD_GS)==0, "A must be divisible by", HAD_GS);

    if(HAD_GS==16){
        fusedQuantizeNvAbsMax_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==32){
        fusedQuantizeNvAbsMaxHad32_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==64){
        fusedQuantizeNvAbsMaxHad64_host(OUT, OUT_sf, A, B, global_scale);
    } else if(HAD_GS==128){
#if TARGET_CUDA_ARCH == 100
        fusedQuantizeNvAbsMax_host_sm100(OUT, OUT_sf, A, B, global_scale);
#elif TARGET_CUDA_ARCH == 120
        fusedQuantizeNvAbsMaxHad128_host(OUT, OUT_sf, A, B, global_scale);
#endif
    } else {
        STD_TORCH_CHECK(false,
                    "Unsupported rotation size ", HAD_GS,
                    "; expected 16, 32, 64, or 128.");
    }

    return std::make_tuple(OUT, OUT_sf);
}

void backward_t_bf16(const torch::stable::Tensor& x,
                     const torch::stable::Tensor& h,
                     torch::stable::Tensor& xh_e2m1,
                     torch::stable::Tensor& xh_e8m0)
{
    int err = backward_t_bf16_cuda(
        x.const_data_ptr(),
        h.const_data_ptr(),
        xh_e2m1.mutable_data_ptr(),
        xh_e8m0.mutable_data_ptr(),
        x.size(-1),
        x.size(-2),
        x.numel() / (x.size(-2) * x.size(-1)),
        get_current_cuda_stream(h.get_device_index())
    );
    (void)err;
}

void backward_qt_bf16(const torch::stable::Tensor& x_e2m1,
                      const torch::stable::Tensor& x_e8m0,
                      const torch::stable::Tensor& h,
                      const torch::stable::Tensor& alpha,
                      torch::stable::Tensor& xh_e2m1,
                      torch::stable::Tensor& xh_e8m0) {
    int err = backward_qt_bf16_cuda(
        x_e2m1.const_data_ptr(),
        x_e8m0.const_data_ptr(),
        h.const_data_ptr(),
        alpha.const_data_ptr(),
        xh_e2m1.mutable_data_ptr(),
        xh_e8m0.mutable_data_ptr(),
        x_e2m1.size(-1) * 2,
        x_e2m1.size(-2),
        x_e2m1.numel() / (x_e2m1.size(-2) * x_e2m1.size(-1)),
        get_current_cuda_stream(h.get_device_index())
    );
    (void)err;
}

void backward_bf16_square_double_mxfp8(const torch::stable::Tensor& x_bf16,
    torch::stable::Tensor& x_fp8,
    torch::stable::Tensor& row_scales,
    torch::stable::Tensor& column_scales) {
    int err = backward_bf16_square_double_mxfp8_cuda(
        x_bf16.const_data_ptr(),
        x_bf16.size(0),
        x_bf16.size(1),
        x_fp8.mutable_data_ptr(),
        row_scales.mutable_data_ptr(),
        column_scales.mutable_data_ptr(),
        get_current_cuda_stream(x_bf16.get_device_index())
    );
    (void)err;
}

void mxfp4_transpose_mxfp8(const torch::stable::Tensor& x_fp4,
    const torch::stable::Tensor& scales,
    torch::stable::Tensor& x_fp8,
    torch::stable::Tensor& shared_exps) {
    int err = mxfp4_transpose_mxfp8_cuda(
        x_fp4.const_data_ptr(),
        scales.const_data_ptr(),
        x_fp4.size(0),
        x_fp4.size(1) * 2,
        x_fp8.mutable_data_ptr(),
        shared_exps.mutable_data_ptr(),
        get_current_cuda_stream(x_fp4.get_device_index())
    );
    (void)err;
}


STABLE_TORCH_LIBRARY_FRAGMENT(_qutlass_C, m) {
  m.def("matmul_mxf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
  m.def("matmul_nvf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
  m.def("matmul_ada_mxf4_bf16_tn(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");

  m.def("fusedQuantizeMxQuest(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf) -> (Tensor, Tensor)");
  m.def("fusedQuantizeMxQuestWithMask(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf, Tensor OUT_mask) -> (Tensor, Tensor, Tensor)");
  m.def("fusedQuantizeMxAbsMax(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf) -> (Tensor, Tensor)");
  m.def("fusedQuantizeNvQuest(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf, Tensor global_scale) -> (Tensor, Tensor)");
  m.def("fusedQuantizeNvAbsMax(Tensor A, Tensor R, Tensor OUT, Tensor OUT_sf, Tensor global_scale) -> (Tensor, Tensor)");

  //m.def("backward_t_bf16(Tensor x_e2m1, Tensor x_e8m0, Tensor h, float alpha, Tensor xh_e2m1, Tensor xh_e8m0) -> void");
  //m.def("backward_qt_bf16(Tensor x, Tensor h, Tensor xh_e2m1, Tensor xh_e8m0) -> void");
}

STABLE_TORCH_LIBRARY_IMPL(_qutlass_C, CUDA, m) {
  m.impl("matmul_mxf4_bf16_tn",      TORCH_BOX(&QUTLASS::matmul_mxf4_bf16_tn));
  m.impl("matmul_nvf4_bf16_tn",      TORCH_BOX(&QUTLASS::matmul_nvf4_bf16_tn));
  m.impl("matmul_ada_mxf4_bf16_tn",  TORCH_BOX(&QUTLASS::matmul_ada_mxf4_bf16_tn));

  m.impl("fusedQuantizeMxQuest",     TORCH_BOX(&QUTLASS::fusedQuantizeMxQuest));
  m.impl("fusedQuantizeMxQuestWithMask", TORCH_BOX(&QUTLASS::fusedQuantizeMxQuestWithMask));
  m.impl("fusedQuantizeMxAbsMax",    TORCH_BOX(&QUTLASS::fusedQuantizeMxAbsMax));
  m.impl("fusedQuantizeNvQuest",     TORCH_BOX(&QUTLASS::fusedQuantizeNvQuest));
  m.impl("fusedQuantizeNvAbsMax",    TORCH_BOX(&QUTLASS::fusedQuantizeNvAbsMax));

  //m.impl("backward_t_bf16",          TORCH_BOX(&QUTLASS::backward_t_bf16));
  //m.impl("backward_qt_bf16",         TORCH_BOX(&QUTLASS::backward_qt_bf16));
}

}