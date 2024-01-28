#pragma once
#include <pybind11/pybind11.h>

namespace TORCHQ::asymmetric {
void buildSubmodule(pybind11::module &mod);
}  // namespace TORCHQ::asymmetric
