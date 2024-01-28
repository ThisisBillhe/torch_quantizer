# Torch Quantizer

`torch_quantizer` is a Python package designed for efficient quantization of PyTorch models, particularly focusing on converting floating-point Linear and Conv2d modules to INT8 precision for improved inference speed on CUDA backend. Also, torch_quantizer supports [temporary quantization](https://arxiv.org/pdf/2310.03270) and is specially optimized for diffusion models.

## Installation

Before installing `torch_quantizer`, ensure you have PyTorch installed in your environment.

To install pre-built `torch_quantizer`, use the following command:

```bash
pip install torch_quantizer-*.whl
```
To build from source, cloning the repository and compiling with NVCC:
```bash
pip install -e .
```
## Usage

### Prerequisites

Import PyTorch before importing `torch_quantizer`:

```python
import torch
import torch_quantizer as tq
```

### Benchmarking

To benchmark and verify the inference speedup by INT8 operation, use the following methods:

```bash
python3 example/benchmark/benchmark.py --o linear --bs 512 --cin 960 --cout 960
python3 example/benchmark/benchmark.py --o conv2d --bs 1 --cin 512 --cout 512
```
#### Benchmarking Results

Here are the results for both linear and 2D convolution operations, showing the average time taken for FP32, FP16, and INT8 (Quant+Dequant) operations on RTX 3090.

##### Linear Operation

- **Batch Size:** 512
- **Channels In:** 960
- **Channels Out:** 960

| Operation Type | Average Time    |
|----------------|-----------------|
| FP32           | 8.414e-05 s     |
| FP16           | 3.304e-05 s     |
| INT8 (Quant+Dequant) | 2.908e-05 s |

##### 2D Convolution Operation

- **Batch Size:** 1
- **Channels In:** 512
- **Channels Out:** 512

| Operation Type | Average Time    |
|----------------|-----------------|
| FP32           | 0.000903 s      |
| FP16           | 0.000413 s      |
| INT8 (Quant+Dequant) | 0.000178 s |

These results highlight the performance improvements achieved through quantization to INT8, demonstrating significant reductions in inference time across both operations.

### Model Conversion to INT8

#### Step 1: FP16 Precision Check

Ensure your model can be inferenced with FP16 precision. For instance:

```python
unet.half()
# do FP16 inference
```

#### Step 2: Fake Quantization

Convert your model to a fake quantized model for calibration:

```python
wq_params = {'n_bits': 8, 'channel_wise': True, 'scale_method': 'max'}
aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse'}
ddim_steps = # Define diffusion steps here, 1 for non-temporal models

fakeq_model = tq.fake_quant(unet, wq_params, aq_params, num_steps=ddim_steps)
```

#### Step 3: Calibration

Run your fake quantized model with some input for calibration. For example:

```python
from some_diffusion_library import DiffusionPipeline

pipe = DiffusionPipeline()
prompt = "a photo of a flying dog"
image = pipe(prompt, guidance_scale=7.5)["sample"][0]
```

#### Step 4: Conversion to Real INT8 Model

Convert the fake quantized model to a real INT8 model:

```python
qunet = tq.fake2real(fakeq_model, save_dir='.')
```

The INT8 checkpoint will be saved in the specified directory.

### Loading INT8 Model

Load the INT8 model directly from the checkpoint:

```python
ckpt_path = 'path_to_checkpoint'
qnn = tq.real_quant(unet, n_bits, ddim_steps, ckpt_path)
```

## Acknowledgement

This repository is built upon [QUIK](https://github.com/IST-DASLab/QUIK). We thank the authors for their open-sourced code.

## License

`torch_quantizer` is released under [Apache-2.0 License](LICENSE).
