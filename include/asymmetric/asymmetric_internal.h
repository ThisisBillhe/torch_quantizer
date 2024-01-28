#pragma once
#include <torch/extension.h>

namespace TORCHQ::asymmetric {

torch::Tensor myQuantizeCUDA(const torch::Tensor &src, const torch::Tensor &delta,
                              const torch::Tensor &zp);

torch::Tensor myQuantizeNCHWCUDA(const torch::Tensor &src, const torch::Tensor &delta,
                              const torch::Tensor &zp);
                              
}  // namespace TORCHQ::asymmetric