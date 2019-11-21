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

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
__global__ void KeGroupPointsFW(int b, int c, int n, int npoints, int nsample,
                                const T* __restrict__ points,
                                const int* __restrict__ idx,
                                T* __restrict__ out) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_idx = index / nsample;
  if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

  int sample_idx = index % nsample;

  idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;
  int in_idx = bs_idx * c * n + c_idx * n + idx[0];
  int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
                pt_idx * nsample + sample_idx;

  out[out_idx] = points[in_idx];
}

template <typename T>

__global__ void KeGroupPointsBW(int b, int c, int n, int npoints, int nsample,
                                const T* __restrict__ grad_out,
                                const int* __restrict__ idx,
                                T* __restrict__ grad_points) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_idx = index / nsample;
  if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

  int sample_idx = index % nsample;
  grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
              pt_idx * nsample + sample_idx;
  idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;

  platform::CudaAtomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0],
                          grad_out[0]);
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
    const int c = input->dims()[1];
    const int n = input->dims()[2];
    const int m = idx->dims()[1];
    const int s = idx->dims()[2];

    auto* output_data = output->mutable_data<T>({b, c, m, s}, ctx.GetPlace());

    dim3 blocks(DIVUP(m * s, THREADS_PER_BLOCK), c, b);
    dim3 threads(THREADS_PER_BLOCK);
    KeGroupPointsFW<
        T><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
        b, c, n, m, s, input_data, idx_data, output_data);
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
    const int c = input->dims()[1];
    const int n = input->dims()[2];
    const int m = idx->dims()[1];
    const int s = idx->dims()[2];

    auto* input_grad_data =
        input_grad->mutable_data<T>({b, c, n}, ctx.GetPlace());
    auto& dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    int pnum = input_grad->numel();
    Zero<<<(pnum + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(input_grad_data,
                                                               pnum);

    dim3 blocks(DIVUP(m * s, THREADS_PER_BLOCK), c, b);
    dim3 threads(THREADS_PER_BLOCK);

    KeGroupPointsBW<<<blocks, threads, 0, dev_ctx.stream()>>>(
        b, c, n, m, s, output_grad_data, idx_data, input_grad_data);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(group_points, ops::GroupPointsOpCUDAKernel<float>,
                        ops::GroupPointsOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(group_points_grad,
                        ops::GroupPointsGradOpCUDAKernel<float>,
                        ops::GroupPointsGradOpCUDAKernel<double>);
