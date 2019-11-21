/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cuda_primitives.h"

#include "util.cu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void GatherPointKernel(int b,
                                  int n,
                                  int m,
                                  const T *__restrict__ inp,
                                  const int *__restrict__ idx,
                                  T *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < m;
         j += blockDim.x * gridDim.y) {
      int a = idx[i * m + j];
      for (int k = 0; k < 3; k++) {
        out[(i * m + j) * 3 + k] = inp[(i * n + a) * 3 + k];
      }
    }
  }
}

template <typename T>
__global__ void GatherPointGradKernel(int b,
                                      int n,
                                      int m,
                                      const T *__restrict__ out_grad,
                                      const int *__restrict__ idx,
                                      T *__restrict__ in_grad) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < m;
         j += blockDim.x * gridDim.y) {
      int a = idx[i * m + j];
      const T *out_grad_pos = &out_grad[(i * m + j) * 3];
      T *in_grad_pos = &in_grad[(i * n + a) * 3];
      for (int k = 0; k < 3; k++) {
        platform::CudaAtomicAdd(&in_grad_pos[k], out_grad_pos[k]);
      }
    }
  }
}

template <typename T>
class GatherPointOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *points = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *output = ctx.Output<Tensor>("Output");

    if (points->numel() == 0) return;

    const T *p_points = points->data<T>();
    const int *p_index = index->data<int>();
    T *p_out_points = output->mutable_data<T>(ctx.GetPlace());

    int batch_size = points->dims()[0];
    int n_points = points->dims()[1];
    int m_points = index->dims()[1];

    GatherPointKernel<<<dim3(2, 8, 1), 512>>>(
        batch_size, n_points, m_points, p_points, p_index, p_out_points);
  }
};

template <typename T>
class GatherPointGradOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *points = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto *points_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    if (points->numel() == 0) return;

    const T *p_output_grad = output_grad->data<T>();
    const int *p_index = index->data<int>();
    T *p_points_grad = points_grad->mutable_data<T>(ctx.GetPlace());
    int pnum = points_grad->numel();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    Zero<<<(pnum + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(p_points_grad,
                                                               pnum);

    int batch_size = points->dims()[0];
    int n_points = points->dims()[1];
    int m_points = index->dims()[1];

    GatherPointGradKernel<<<dim3(2, 8, 1), 512, 0, dev_ctx.stream()>>>(
        batch_size, n_points, m_points, p_output_grad, p_index, p_points_grad);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(gather_point,
                        ops::GatherPointOpCUDAKernel<float>,
                        ops::GatherPointOpCUDAKernel<double>,
                        ops::GatherPointOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(gather_point_grad,
                        ops::GatherPointGradOpCUDAKernel<float>,
                        ops::GatherPointGradOpCUDAKernel<double>,
                        ops::GatherPointGradOpCUDAKernel<int>);
