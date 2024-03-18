#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/device_nhwc_to_nchw.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include "int4.h"
#include "matmul/matmul_internal.h"
#include "util.h"


namespace TORCHQ::matmul {
torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 128>,
      cutlass::gemm::GemmShape<32, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;
      
  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;


  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

template <typename KTorch>
__global__ void myDequantizationKernel(KTorch *__restrict__ out,
                                     const int *__restrict__ x,
                                     const KTorch *__restrict__ zp_times_weight_channel_sum,
                                     const KTorch *__restrict__ act_times_weight_delta,
                                     const KTorch *__restrict__ y,
                                     const unsigned rows, const unsigned cols) {
    const unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= cols || row >= rows) {
        return;
    }
    // using K = typename util::DtypeTorchDispatcher<KTorch>::value;
    // Convert int32_t element to float32 first
    float xElement = static_cast<float>(x[col + row * cols]);
    // float zp_times_weight_channel_sum_element = util::type2float(zp_times_weight_channel_sum[row]);
    // float act_times_weight_delta_element = util::type2float(act_times_weight_delta[row]);

    float zp_times_weight_channel_sum_element = zp_times_weight_channel_sum[col];
    float act_times_weight_delta_element = act_times_weight_delta[col];

    // Subtract zp_times_weight_channel_sum and multiply by act_times_weight_delta
    xElement -= zp_times_weight_channel_sum_element;
    xElement *= act_times_weight_delta_element;
    xElement += y[col];

    // out[col + row * cols] = util::float2type<K>(xElement);

    out[col + row * cols] = xElement;
}

torch::Tensor myInt8MatmulCUDA(const torch::Tensor &A, 
                                         const torch::Tensor &B,
                                         const torch::Tensor & zp_times_weight_channel_sum,
                                         const torch::Tensor & act_times_weight_delta,
                                         const torch::Tensor & y) {
    // Step 1: Perform GEMM Operation
    torch::Tensor C = int8MatmulCUDA(A, B);

    // Step 2: Setup for Dequantization
    unsigned M = C.size(0);
    unsigned N = C.size(1);
    auto out = torch::empty_like(C, torch::dtype(torch::kFloat).device(C.device()));

    // Step 3: Dequantization Kernel Call
    dim3 blockDim(16, 16);  // Adjust block size as needed, how many thread in a block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    myDequantizationKernel<<<gridDim, blockDim>>>(out.data_ptr<float>(),
                                                C.data_ptr<int>(),
                                                zp_times_weight_channel_sum.data_ptr<float>(),
                                                act_times_weight_delta.data_ptr<float>(),
                                                y.data_ptr<float>(),
                                                M, N);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle the error appropriately
    }

    // Step 4: Return the dequantized output
    return out;
}

template <typename KTorch>
__global__ void myDequantizationConvKernel(KTorch *__restrict__ out,
                                           const int *__restrict__ x,
                                           const KTorch *__restrict__ zp_times_weight_channel_sum,
                                           const KTorch *__restrict__ act_times_weight_delta,
                                           const KTorch *__restrict__ y,
                                           const unsigned N, const unsigned H, const unsigned W, const unsigned C) {
    // Calculate indices for H, W, and C using block dimensions
    const unsigned nhw = blockIdx.x * blockDim.x + threadIdx.x; // Index for H
    const unsigned c = blockIdx.y * blockDim.y + threadIdx.y; // Index for W
    // const unsigned c = blockIdx.z * blockDim.z + threadIdx.z; // Index for C

    // Check if the current thread is within the bounds for H, W, and C
    if (nhw >= N*H*W || c >= C) {
        return;
    }

    // Split the combined index back into N and H indices
    // const unsigned n = nh / H; // Recover N
    // const unsigned h = nh % H; // Recover H

    // Calculate the linear index for the 4D array
    // unsigned index = n * H * W * C + h * W * C + w * C + c;
    unsigned index = nhw * C + c;

    // Convert int32_t element to float32
    float xElement = static_cast<float>(x[index]);

    // Get zp_times_weight_channel_sum and act_times_weight_delta for the current channel
    float zp_times_weight_channel_sum_element = zp_times_weight_channel_sum[c];
    float act_times_weight_delta_element = act_times_weight_delta[c];

    xElement -= zp_times_weight_channel_sum_element;
    xElement *= act_times_weight_delta_element;
    xElement += y[c];

    // Write the result back
    out[index] = xElement;

}

__global__ void relu_kernel(int32_t* output, int N, int outputH, int outputW, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * outputH * outputW * C) {
        output[idx] = max(0, output[idx]);
    }
}

torch::Tensor int8ConvCUDA(const torch::Tensor &input, const torch::Tensor &filter, const int padH, const int padW,
                               const int strideH, const int strideW, const int dilationH, const int dilationW) {
  // Check that tensors are on the same GPU
  torch::checkAllSameGPU("int8ConvCUDA", {{input, "input", 0}, {filter, "filter", 1}});

  // Assuming input tensor layout is NCHW and filter layout is KCRS
  auto N = input.size(0);
  auto H = input.size(1);
  auto W = input.size(2);
  auto C = input.size(3);
  auto K = filter.size(0);
  auto R = filter.size(1);
  auto S = filter.size(2);

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = int8_t;
  using ElementB           = int8_t;
  using ElementC           = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute     = float;

  using Conv2dFpropKernel = cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  // Define arguments for CUTLASS Convolution
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;


  // Define output tensor
  // Assuming stride of 1, no padding, and no dilation for simplicity
  auto outputH = (H + padH + padH - R) / strideH + 1;
  auto outputW = (W + padW + padW - S) / strideW + 1;

  torch::Tensor output = torch::empty({N, outputH, outputW, K}, torch::dtype(torch::kInt32).device(input.device()));

  auto output_size = cutlass::Tensor4DCoord(
      N,
      outputH,
      outputW,
      K);

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(
      cutlass::Tensor4DCoord({N, H, W, C}),
      cutlass::Tensor4DCoord({K, R, S, C}),
      cutlass::Tensor4DCoord({padH, padH, padW, padW}),
      cutlass::MatrixCoord({strideH, strideW}),
      cutlass::MatrixCoord({dilationH, dilationW}),
      output_size,
      mode,
      split_k_slices
  );
  
  cutlass::TensorRef<int8_t, cutlass::layout::TensorNHWC> input_ref(
    input.data_ptr<int8_t>(), cutlass::layout::TensorNHWC::packed(cutlass::Tensor4DCoord({N, H, W, C})));

  cutlass::TensorRef<int8_t, cutlass::layout::TensorNHWC> filter_ref(
    filter.data_ptr<int8_t>(), cutlass::layout::TensorNHWC::packed(cutlass::Tensor4DCoord({K, R, S, C})));

  cutlass::TensorRef<int32_t, cutlass::layout::TensorNHWC> output_ref(
    output.data_ptr<int32_t>(), cutlass::layout::TensorNHWC::packed(cutlass::Tensor4DCoord({N, outputH, outputW, K})));

  // Set up arguments for the convolution operation
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    input_ref,
    filter_ref,
    output_ref,
    output_ref,
    {1, 0}};

  // Launch the convolution
  ImplicitGemm convOp;
  auto status = convOp(arguments);
  
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));

  return output;
}

torch::Tensor myInt8ConvCUDA(const torch::Tensor &input, const torch::Tensor &filter, const int padH, const int padW,
                               const int strideH, const int strideW, const int dilationH, const int dilationW,
                                         const torch::Tensor & zp_times_weight_channel_sum,
                                         const torch::Tensor & act_times_weight_delta,
                                         const torch::Tensor & y, const bool relu_fushion) {
    // Step 1: Perform GEMM Operation
    torch::Tensor C = int8ConvCUDA(input, filter, padH, padW, strideH, strideW, dilationH, dilationW);
    auto N = C.size(0);
    auto H = C.size(1);
    auto W = C.size(2);
    auto Co = C.size(3);

    // Step 1.5: Layer fushion: Conv + RELU
    if (relu_fushion) {
      int threadsPerBlockRELU = 256;
      int blocksPerGridRELU = (N * H * W * Co + threadsPerBlockRELU - 1) / threadsPerBlockRELU;
      relu_kernel<<<blocksPerGridRELU, threadsPerBlockRELU>>>(C.data_ptr<int32_t>(), N, H, W, Co);
    }

    // Step 2: Setup for Dequantization
    auto out = torch::empty_like(C, torch::dtype(torch::kFloat).device(C.device()));


    // Step 3: Dequantization Kernel Call
    dim3 threadsPerBlock(16, 16); // Adjust as necessary
    dim3 numBlocks((N*H*W + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (Co + threadsPerBlock.y - 1) / threadsPerBlock.y);
    myDequantizationConvKernel<<<numBlocks, threadsPerBlock>>>(out.data_ptr<float>(), 
                                                              C.data_ptr<int>(),
                                                              zp_times_weight_channel_sum.data_ptr<float>(),
                                                              act_times_weight_delta.data_ptr<float>(),
                                                              y.data_ptr<float>(), N, H, W, Co);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        // Handle the error appropriately
    }

    // Step 4: Return the dequantized output
    // return out;

      // NHWC to NCHW
      torch::Tensor dst_nchw = torch::empty({N, Co, H, W},
                                            torch::dtype(torch::kFloat).device(C.device()));

      // Step 3: Define tensor sizes for NHWC and NCHW
      auto input_tensor_size = cutlass::Tensor4DCoord({N, H, W, Co});
      auto output_tensor_size = cutlass::Tensor4DCoord({N, Co, H, W});

      // Step 4: Create Tensor Refs
      cutlass::TensorRef<float, cutlass::layout::TensorNHWC> ref_input(
        out.data_ptr<float>(), cutlass::layout::TensorNHWC::packed(cutlass::Tensor4DCoord({N,H,W,Co})));
      cutlass::TensorRef<float, cutlass::layout::TensorNCHW> ref_output(
        dst_nchw.data_ptr<float>(), cutlass::layout::TensorNCHW::packed(cutlass::Tensor4DCoord({N,Co,H,W})));

      // Call nchw_to_nhwc
      nhwc_to_nchw(input_tensor_size, output_tensor_size, ref_input, ref_output, 0);  // Assuming default stream

    return dst_nchw;
}
}  // namespace TORCHQ::matmul
