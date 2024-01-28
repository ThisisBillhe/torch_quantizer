#include "asymmetric/asymmetric.h"

#include <torch/extension.h>

#include "asymmetric/asymmetric_internal.h"

namespace TORCHQ::asymmetric {
torch::Tensor myQuantize(const torch::Tensor &src, const torch::Tensor &delta,
                              const torch::Tensor &zp) {
  torch::checkAllContiguous("myQuantize", {{src, "src", 0}, {delta, "delta", 1}, {zp, "zp", 2}});
  torch::checkDeviceType("myQuantize", {src, delta, zp}, at::DeviceType::CUDA);
  return myQuantizeCUDA(src, delta, zp);
}

torch::Tensor myQuantizeNCHW(const torch::Tensor &src, const torch::Tensor &delta,
                              const torch::Tensor &zp) {
  torch::checkAllContiguous("myQuantizeNCHW", {{src, "src", 0}, {delta, "delta", 1}, {zp, "zp", 2}});
  torch::checkDeviceType("myQuantizeNCHW", {src, delta, zp}, at::DeviceType::CUDA);
  return myQuantizeNCHWCUDA(src, delta, zp);
}

void buildSubmodule(py::module &mod) {
  py::module m =
      mod.def_submodule("asymmetric", "Asymmetric Quantization Functions");

  m.def("myQuantize", &myQuantize,
        "input: (src: torch.Tensor(M x N, FP16, CUDA),\n"
        "delta: torch.Tensor(1, FP16, CUDA)\n"
        "zp: torch.Tensor(1, INT8, CUDA) \n"
        "output: torch.Tensor(M x N, INT8, CUDA)\n"
        "output = int{bits}Packing(int{bits}Rounding((source / delta) + zp ",
        py::arg("src"), py::arg("delta"), py::arg("zp"));
  m.def("myQuantizeNCHW", &myQuantizeNCHW,
        "input: (src: torch.Tensor(N x C x H x W, FP16, CUDA),\n"
        "delta: torch.Tensor(1, FP16, CUDA)\n"
        "zp: torch.Tensor(1, INT8, CUDA) \n"
        "output: torch.Tensor(M x N, INT8, CUDA)\n"
        "output = int{bits}Packing(int{bits}Rounding((source / delta) + zp ",
        py::arg("src"), py::arg("delta"), py::arg("zp"));
}
}  // namespace TORCHQ::asymmetric