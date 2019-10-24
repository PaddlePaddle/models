/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cuda_primitives.h"

#include "util.cu.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
__global__ void KeThreeInterpFw(T* output,
                                const T* input,
                                const T* weight,
                                const int* idx,
                                const int b,
                                const int m,
                                const int c,
                                const int n) {
  int nthreads = b * n * c;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int bi = tid / n / c;
    int ni = (tid % (n * c)) / c;
    int ci = tid % c;

    int input_base_idx = bi * m * c;
    int w_idx = bi * n * 3 + ni * 3;
    output[tid] =
        input[input_base_idx + idx[w_idx] * c + ci] * weight[w_idx] +
        input[input_base_idx + idx[w_idx + 1] * c + ci] * weight[w_idx + 1] +
        input[input_base_idx + idx[w_idx + 2] * c + ci] * weight[w_idx + 2];
  }
}

template <typename T>
__global__ void KeThreeInterpBw(T* input_grad,
                                const T* output_grad,
                                const T* weight,
                                const int* idx,
                                const int b,
                                const int m,
                                const int c,
                                const int n) {
  int nthreads = b * n * c;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int bi = tid / n / c;
    int ni = (tid % (c * n)) / c;
    int ci = tid % c;

    int input_base_idx = bi * m * c;
    int w_idx = bi * n * 3 + ni * 3;
    platform::CudaAtomicAdd(&input_grad[input_base_idx + idx[w_idx] * c + ci],
                            output_grad[tid] * weight[w_idx]);
    platform::CudaAtomicAdd(
        &input_grad[input_base_idx + idx[w_idx + 1] * c + ci],
        output_grad[tid] * weight[w_idx + 1]);
    platform::CudaAtomicAdd(
        &input_grad[input_base_idx + idx[w_idx + 2] * c + ci],
        output_grad[tid] * weight[w_idx + 2]);
  }
}

template <typename T>
class ThreeInterpOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* input = ctx.Input<Tensor>("X");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* idx = ctx.Input<Tensor>("Idx");
    auto* output = ctx.Output<Tensor>("Out");
    auto* input_data = input->data<T>();
    auto* weight_data = weight->data<T>();
    auto* idx_data = idx->data<int>();

    const int b = input->dims()[0];
    const int m = input->dims()[1];
    const int c = input->dims()[2];
    const int n = weight->dims()[1];

    auto* output_data = output->mutable_data<T>({b, n, c}, ctx.GetPlace());

    int pixelNum = b * n * c;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeThreeInterpFw<
        T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        output_data, input_data, weight_data, idx_data, b, m, c, n);
  }
};

template <typename T>
class ThreeInterpGradOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* idx = ctx.Input<Tensor>("Idx");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* weight_data = weight->data<T>();
    auto* idx_data = idx->data<int>();
    auto output_grad_data = output_grad->data<T>();

    const int b = input->dims()[0];
    const int m = input->dims()[1];
    const int c = input->dims()[2];
    const int n = weight->dims()[1];

    auto* input_grad_data =
        input_grad->mutable_data<T>({b, m, c}, ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int pnum = input_grad->numel();
    Zero<<<(pnum + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(input_grad_data,
                                                               pnum);

    int pixelNum = b * n * c;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeThreeInterpBw<
        T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        input_grad_data, output_grad_data, weight_data, idx_data, b, m, c, n);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(three_interp,
                        ops::ThreeInterpOpCUDAKernel<float>,
                        ops::ThreeInterpOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(three_interp_grad,
                        ops::ThreeInterpGradOpCUDAKernel<float>,
                        ops::ThreeInterpGradOpCUDAKernel<double>);
