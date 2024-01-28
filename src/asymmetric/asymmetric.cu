#include <c10/cuda/CUDAGuard.h>
#include <cutlass/util/device_nchw_to_nhwc.h>
#include "asymmetric/asymmetric_internal.h"
#include "int4.h"
#include "util.h"

namespace TORCHQ::asymmetric {
const unsigned MAX_NUMTHREADS = 1024;
const unsigned MAX_NUMBER_BLOCKS = 65535;
unsigned NUM_STRIDES_PER_THREAD_QUANTIZE = 0;
unsigned NUM_STRIDES_PER_THREAD_DEQUANTIZE = 0;

__global__ void myQuantizeCUDAKernel8Bits(int8_t *__restrict__ dst,
                                        const torch::Half *__restrict__ src,
                                        const torch::Half * __restrict__ delta,
                                        const torch::Half  * __restrict__ zp,
                                        const unsigned rows,
                                        const unsigned cols) {
  const unsigned thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = blockDim.x * gridDim.x;
  const unsigned num_elems = rows * cols;

  for (unsigned idx = thread_id; idx < num_elems; idx += stride) {
    const unsigned row = idx / cols;
    
    __half data = __hadd(__hdiv(src[idx], delta[0]), zp[0]);
    int val = __half2int_rn(data);
    // needs to be shifted by 128 to fit int8_t
    val = static_cast<int8_t>(min(max(val, -128), 127));
    dst[idx] = val;
  }
}

torch::Tensor myQuantizeCUDA(const torch::Tensor &src, const torch::Tensor &delta,
                              const torch::Tensor &zp) {
  torch::checkAllSameGPU("myQuantizeCUDA", {{src, "src", 0}, {delta, "delta", 1}, {zp, "zp", 2}});
  const at::cuda::CUDAGuard device_guard(src.device());

  if (NUM_STRIDES_PER_THREAD_QUANTIZE == 0) {
    char const *temp = getenv("NUM_STRIDES_PER_THREAD_QUANTIZE");
    if (temp)
      NUM_STRIDES_PER_THREAD_QUANTIZE = std::atoi(temp);
    else
      NUM_STRIDES_PER_THREAD_QUANTIZE = 1;
    TORCH_CHECK(NUM_STRIDES_PER_THREAD_QUANTIZE > 0 and
                    NUM_STRIDES_PER_THREAD_QUANTIZE < 64,
                "Quantize: invalid value of NUM_STRIDES_PER_THREAD_QUANTIZE");
  }

  unsigned rows = src.size(0);
  unsigned colsSrc = src.size(1);
  torch::Tensor dst;
  const unsigned num_elems = src.numel();
  const unsigned num_threads = min(num_elems, MAX_NUMTHREADS);
  const unsigned num_blocks =
      max((num_elems + num_threads - 1) /
              (num_threads * NUM_STRIDES_PER_THREAD_QUANTIZE),
          16);


  dst = torch::empty({rows, colsSrc},
                      torch::dtype(util::TorchDtypeDispatcher<int8_t>::value)
                          .device(src.device()));
  myQuantizeCUDAKernel8Bits<<<num_blocks, num_threads>>>(
      dst.data_ptr<int8_t>(), src.data_ptr<torch::Half>(),
      delta.data_ptr<torch::Half>(), zp.data_ptr<torch::Half>(), rows, colsSrc);
  
  auto status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Failed quantize: " + std::string(cudaGetErrorString(status)));
  return dst;
  }


__global__ void myQuantizeNCHWCUDAKernel8Bits(int8_t *__restrict__ dst,
                                        const torch::Half *__restrict__ src,
                                        const torch::Half * __restrict__ delta,
                                        const torch::Half  * __restrict__ zp,
                                        const unsigned num_elems) {
  const unsigned thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned stride = blockDim.x * gridDim.x;
  // const unsigned num_elems = rows * cols;

  for (unsigned idx = thread_id; idx < num_elems; idx += stride) {
    // const unsigned row = idx / cols;
    
    __half data = __hadd(__hdiv(src[idx], delta[0]), zp[0]);
    int val = __half2int_rn(data);
    // needs to be shifted by 128 to fit int8_t
    val = static_cast<int8_t>(min(max(val, -128), 127));
    dst[idx] = val;
  }
}

torch::Tensor myQuantizeNCHWCUDA(const torch::Tensor &src, const torch::Tensor &delta,
                              const torch::Tensor &zp) {
  torch::checkAllSameGPU("myQuantizeNCHWCUDA", {{src, "src", 0}, {delta, "delta", 1}, {zp, "zp", 2}});
  const at::cuda::CUDAGuard device_guard(src.device());

  if (NUM_STRIDES_PER_THREAD_QUANTIZE == 0) {
    char const *temp = getenv("NUM_STRIDES_PER_THREAD_QUANTIZE");
    if (temp)
      NUM_STRIDES_PER_THREAD_QUANTIZE = std::atoi(temp);
    else
      NUM_STRIDES_PER_THREAD_QUANTIZE = 1;
    TORCH_CHECK(NUM_STRIDES_PER_THREAD_QUANTIZE > 0 and
                    NUM_STRIDES_PER_THREAD_QUANTIZE < 64,
                "Quantize: invalid value of NUM_STRIDES_PER_THREAD_QUANTIZE");
  }

  auto N = src.size(0);
  auto C = src.size(1);
  auto H = src.size(2);
  auto W = src.size(3);

  torch::Tensor dst;
  const unsigned num_elems = src.numel();
  const unsigned num_threads = min(num_elems, MAX_NUMTHREADS);
  const unsigned num_blocks =
      max((num_elems + num_threads - 1) /
              (num_threads * NUM_STRIDES_PER_THREAD_QUANTIZE),
          16);


  dst = torch::empty({N,C,H,W},
                      torch::dtype(util::TorchDtypeDispatcher<int8_t>::value)
                          .device(src.device()));
  myQuantizeNCHWCUDAKernel8Bits<<<num_blocks, num_threads>>>(
      dst.data_ptr<int8_t>(), src.data_ptr<torch::Half>(),
      delta.data_ptr<torch::Half>(), zp.data_ptr<torch::Half>(), num_elems);
  
  auto status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "Failed quantize: " + std::string(cudaGetErrorString(status)));

    // NCHW to NHWC
    torch::Tensor dst_nhwc = torch::empty({N, H, W, C},
                                          torch::dtype(util::TorchDtypeDispatcher<int8_t>::value)
                                              .device(src.device()));

    // Step 3: Define tensor sizes for NHWC and NCHW
    auto input_tensor_size = cutlass::Tensor4DCoord({N, C, H, W});
    auto output_tensor_size = cutlass::Tensor4DCoord({N, H, W, C});

    // Step 4: Create Tensor Refs
    cutlass::TensorRef<int8_t, cutlass::layout::TensorNCHW> ref_input(
      dst.data_ptr<int8_t>(), cutlass::layout::TensorNCHW::packed(cutlass::Tensor4DCoord({N,C,H,W})));
    cutlass::TensorRef<int8_t, cutlass::layout::TensorNHWC> ref_output(
      dst_nhwc.data_ptr<int8_t>(), cutlass::layout::TensorNHWC::packed(cutlass::Tensor4DCoord({N,H,W,C})));

    // Call nchw_to_nhwc
    nchw_to_nhwc(input_tensor_size, output_tensor_size, ref_input, ref_output, 0);  // Assuming default stream

  return dst_nhwc;        
  // return dst;
  }


}  // namespace TORCHQ::asymmetric
