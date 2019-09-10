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
__global__ void KeGroupPointsFw(T* output,
                                const T* input,
                                const int* idx,
                                const int b,
                                const int n,
                                const int c,
                                const int ms) {
  int nthreads = b * ms;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int bi = tid / ms;

    int input_base_idx = bi * n * c;
    for (int i = 0; i < c; i++) {
      output[tid * c + i] = input[input_base_idx + idx[tid] * c + i];
    }
  }
}

template <typename T>
__global__ void KeGroupPointsBw(T* input_grad,
                                const T* output_grad,
                                const int* idx,
                                const int b,
                                const int n,
                                const int c,
                                const int ms) {
  int nthreads = b * ms;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int bi = tid / ms;

    int input_base_idx = bi * n * c;
    for (int i = 0; i < c; i++) {
      platform::CudaAtomicAdd(&input_grad[input_base_idx + idx[tid] * c + i],
                              output_grad[tid * c + i]);
    }
  }
}

template <typename T>
class GroupPointsOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* input = ctx.Input<Tensor>("X");
    auto* idx = ctx.Input<Tensor>("Idx");
    auto* output = ctx.Output<Tensor>("Out");
    auto* input_data = input->data<T>();
    auto* idx_data = idx->data<int>();

    const int b = input->dims()[0];
    const int n = input->dims()[1];
    const int c = input->dims()[2];
    const int m = idx->dims()[1];
    const int s = idx->dims()[2];

    auto* output_data = output->mutable_data<T>({b, m, s, c}, ctx.GetPlace());

    const int ms = m * s;
    int pixelNum = b * ms;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeGroupPointsFw<
        T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        output_data, input_data, idx_data, b, n, c, ms);
  }
};

template <typename T>
class GroupPointsGradOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* idx = ctx.Input<Tensor>("Idx");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* idx_data = idx->data<int>();
    auto output_grad_data = output_grad->data<T>();

    const int b = input->dims()[0];
    const int n = input->dims()[1];
    const int c = input->dims()[2];
    const int m = idx->dims()[1];
    const int s = idx->dims()[2];

    auto* input_grad_data =
        input_grad->mutable_data<T>({b, n, c}, ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int pnum = input_grad->numel();
    Zero<<<(pnum + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(input_grad_data,
                                                               pnum);

    const int ms = m * s;
    int pixelNum = b * ms;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeGroupPointsBw<T><<<grid_dim, 512, 0, dev_ctx.stream()>>>(
        input_grad_data, output_grad_data, idx_data, b, n, c, ms);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(group_points,
                        ops::GroupPointsOpCUDAKernel<float>,
                        ops::GroupPointsOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(group_points_grad,
                        ops::GroupPointsGradOpCUDAKernel<float>,
                        ops::GroupPointsGradOpCUDAKernel<double>);
