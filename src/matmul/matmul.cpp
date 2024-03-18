#include "matmul/matmul.h"

#include <torch/extension.h>

#include "matmul/matmul_internal.h"

namespace TORCHQ::matmul {
torch::Tensor int8Matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8Matmul", {A, B}, at::DeviceType::CUDA);
  return int8MatmulCUDA(A, B);
}

torch::Tensor myInt8Matmul(const torch::Tensor &A, 
                                         const torch::Tensor &B,
                                         const torch::Tensor & zp_times_weight_channel_sum,
                                         const torch::Tensor & act_times_weight_delta,
                                         const torch::Tensor & y) {
  torch::checkAllContiguous("myInt8Matmul", {{A, "A", 0}, {B, "B", 1}, {zp_times_weight_channel_sum, "zp_times_weight_channel_sum", 2},
                                               {act_times_weight_delta, "act_times_weight_delta", 3}, {y, "y", 4}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("myInt8Matmul", {A, B, zp_times_weight_channel_sum, act_times_weight_delta, y}, at::DeviceType::CUDA);
  return myInt8MatmulCUDA(A, B, zp_times_weight_channel_sum, act_times_weight_delta, y);
}

torch::Tensor int8Conv(const torch::Tensor &input, const torch::Tensor &filter, const int padH, const int padW,
                               const int strideH, const int strideW, const int dilationH, const int dilationW) {
  torch::checkAllContiguous("int8Conv", {{input, "input", 0}, {filter, "filter", 1}});
  // TODO(Tingxuan): support more data type
  torch::checkDeviceType("int8Conv", {input, filter}, at::DeviceType::CUDA);
  return int8ConvCUDA(input, filter, padH, padW, strideH, strideW, dilationH, dilationW);

}
torch::Tensor myInt8Conv(const torch::Tensor &input, const torch::Tensor &filter, const int padH, const int padW,
                               const int strideH, const int strideW, const int dilationH, const int dilationW,
                                         const torch::Tensor & zp_times_weight_channel_sum,
                                         const torch::Tensor & act_times_weight_delta,
                                         const torch::Tensor & y, const bool relu_fushion) {
  torch::checkAllContiguous("myInt8Conv", {{input, "input", 0}, {filter, "filter", 1}, {zp_times_weight_channel_sum, "zp_times_weight_channel_sum", 2},
                                               {act_times_weight_delta, "act_times_weight_delta", 3}, {y, "y", 4}});
  // TODO(Tingxuan): support more data type
//   std::cout<<relu_fushion<<std::endl;
  torch::checkDeviceType("myInt8Conv", {input, filter, zp_times_weight_channel_sum, act_times_weight_delta, y}, at::DeviceType::CUDA);
  return myInt8ConvCUDA(input, filter, padH, padW, strideH, strideW, dilationH, dilationW,
                             zp_times_weight_channel_sum, act_times_weight_delta, y, relu_fushion);
}

void buildSubmodule(py::module &mod) {
  py::module m = mod.def_submodule("matmul", "Matmul Functions");
  m.def("int8Matmul", &int8Matmul,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(M x N, INT32, CUDA)\n"
        "output = A @ B^T",
        py::arg("A"), py::arg("B"));

  m.def("myInt8Matmul", &myInt8Matmul,
        "input: (A: torch.Tensor(M x K, INT8, CUDA), B: torch.Tensor(N x K, "
        "INT8, CUDA)), zp_times_weight_channel_sum: torch.Tensor(N, INT8, CUDA), act_times_weight_delta: torch.Tensor(N, INT8, CUDA),"
        "y: torch.Tensor(M x K, FP16, CUDA)\n"
        "output: torch.Tensor(M x N, INT32, CUDA).sub_(zp_times_weight_channel_sum).mul_(act_times_weight_delta)+ y\n"
        "output = (A @ B^T - zp_times_weight_channel_sum) * act_times_weight_delta + y",
        py::arg("A"), py::arg("B"), py::arg("zp_times_weight_channel_sum"), py::arg("act_times_weight_delta"), py::arg("y"));

  m.def("int8Conv", &int8Conv,
        "input: (input: torch.Tensor(N x H x W x Cin, INT8, CUDA), filter: torch.Tensor(Co x Cin x K x K, "
        "INT8, CUDA))\n"
        "output: torch.Tensor(N x H' x W'x Co, INT32, CUDA)\n"
        "output = conv(input, filter)",
        py::arg("input"), py::arg("filter"), py::arg("padH"), py::arg("padW"), py::arg("strideH"), py::arg("strideW"), py::arg("dilationH"), py::arg("dilationW"));

  m.def("myInt8Conv", &myInt8Conv,
        "input: (input: torch.Tensor(N x H x W x Cin, INT8, CUDA), filter: torch.Tensor(Co x Cin x K x K, "
        "INT8, CUDA)), zp_times_weight_channel_sum: torch.Tensor(N, INT8, CUDA), act_times_weight_delta: torch.Tensor(N, INT8, CUDA),"
        "y: torch.Tensor(M x K, FP16, CUDA)\n"
        "output: torch.Tensor(N x H' x W'x Co, INT32, CUDA).sub_(zp_times_weight_channel_sum).mul_(act_times_weight_delta)+ y\n"
        "output = conv(input, filter)",
        py::arg("input"), py::arg("filter"), py::arg("padH"), py::arg("padW"), py::arg("strideH"), py::arg("strideW"), 
        py::arg("dilationH"), py::arg("dilationW"), py::arg("zp_times_weight_channel_sum"), py::arg("act_times_weight_delta"), py::arg("y"), py::arg("relu_fushion"));
}
}  // namespace TORCHQ::matmul
