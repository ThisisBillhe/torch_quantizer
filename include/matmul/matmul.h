#pragma once
#include <pybind11/pybind11.h>

namespace TORCHQ::matmul {
void buildSubmodule(pybind11::module &mod);
}
