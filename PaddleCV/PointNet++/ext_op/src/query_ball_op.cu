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

#include "util.cu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
// input: radius (1), nsample (1), points (b,n,3), new_points (b,m,3)
// output: idx (b,m,nsample)
__global__ void QueryBall(int b,
                          int n,
                          int m,
                          T radius,
                          int nsample,
                          const T *points,
                          const T *new_points,
                          int *idx) {
  int batch_index = blockIdx.x;
  points += n * 3 * batch_index;
  new_points += m * 3 * batch_index;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int j = index; j < m; j += stride) {
    int cnt = 0;
    for (int k = 0; k < n; ++k) {
      if (cnt == nsample)
        break;  // only pick the FIRST nsample points in the ball
      float x2 = new_points[j * 3 + 0];
      float y2 = new_points[j * 3 + 1];
      float z2 = new_points[j * 3 + 2];
      float x1 = points[k * 3 + 0];
      float y1 = points[k * 3 + 1];
      float z1 = points[k * 3 + 2];
      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      if (d < radius * radius) {
        if (cnt == 0) {  // set ALL indices to k, s.t. if there are less points
                         // in ball than nsample, we still have valid
                         // (repeating) indices
          for (int l = 0; l < nsample; ++l) idx[j * nsample + l] = k;
        }
        idx[j * nsample + cnt] = k;
        cnt += 1;
      }
    }
  }
}

template <typename T>
class QueryBallOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    // input: radius (1), nsample (1), points (b,n,3), new_points (b,m,3)
    // output: idx (b,m,nsample)
    auto *points = ctx.Input<Tensor>("Points");
    auto *new_points = ctx.Input<Tensor>("New_Points");
    auto *output = ctx.Output<Tensor>("Output");

    float radius = ctx.Attr<T>("Radius");
    int nsample = ctx.Attr<int>("N_sample");

    if (points->numel() == 0 || new_points->numel() == 0) return;

    int batch_size = points->dims()[0];
    int n = points->dims()[1];
    int m = new_points->dims()[1];
    // allocate memory
    int* p_out_points = output->mutable_data<int>({batch_size, m, nsample}, ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int pnum = output->numel();
    Zero<int><<<(pnum + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(p_out_points,
                                                               pnum);

    const T *p_points = points->data<T>();
    const T *p_new_points = new_points->data<T>();

    QueryBall<<<batch_size, 256>>>(batch_size,
                                   n,
                                   m,
                                   radius,
                                   nsample,
                                   p_points,
                                   p_new_points,
                                   p_out_points);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(query_ball, ops::QueryBallOpCUDAKernel<float>);
