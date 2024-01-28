from torch_quantizer._C import matmul, asymmetric
from torch_quantizer.src.converter import fake_quant, real_quant, fake2real
from torch_quantizer.src.benchmark import benchmark_conv2d, benchmark_linear
from torch_quantizer.version import __version__

__all__ = ['matmul', 'asymmetric', 'fake_quant', 'real_quant', 'fake2real', 'benchmark_conv2d' , 'benchmark_linear']