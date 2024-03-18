#pragma once

#include <torch/extension.h>

namespace TORCHQ::matmul {

torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor myInt8MatmulCUDA(const torch::Tensor &A, 
                                         const torch::Tensor &B,
                                         const torch::Tensor & zp_times_weight_channel_sum,
                                         const torch::Tensor & act_times_weight_delta,
                                         const torch::Tensor & y);
                                         
torch::Tensor int8ConvCUDA(const torch::Tensor &input, const torch::Tensor &filter, const int padH, const int padW,
                               const int strideH, const int strideW, const int dilationH, const int dilationW);

torch::Tensor myInt8ConvCUDA(const torch::Tensor &input, const torch::Tensor &filter, const int padH, const int padW,
                               const int strideH, const int strideW, const int dilationH, const int dilationW,
                                         const torch::Tensor & zp_times_weight_channel_sum,
                                         const torch::Tensor & act_times_weight_delta,
                                         const torch::Tensor & y, const bool relu_fushion);
}  // namespace TORCHQ::matmul
