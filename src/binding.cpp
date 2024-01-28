#include <pybind11/pybind11.h>

#include "asymmetric/asymmetric.h"
#include "matmul/matmul.h"

PYBIND11_MODULE(_C, mod) {
  TORCHQ::matmul::buildSubmodule(mod);
  TORCHQ::asymmetric::buildSubmodule(mod);
}
