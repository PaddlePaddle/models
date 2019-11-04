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

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
__global__ void KeThreeNNFw(T* distance,
                            int* idx,
                            const T* input,
                            const T* known,
                            const float eps,
                            const int b,
                            const int n,
                            const int m) {
  int nthreads = b * n;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int bi = tid / n;
    int ni = tid % n;

    int input_idx = tid * 3;
    T x1 = input[input_idx];
    T y1 = input[input_idx + 1];
    T z1 = input[input_idx + 2];

    distance[input_idx] = 1e40;
    distance[input_idx + 1] = 1e40;
    distance[input_idx + 2] = 1e40;
    idx[input_idx] = 0;
    idx[input_idx + 1] = 0;
    idx[input_idx + 2] = 0;
    for (int i = 0; i < m; i++) {
      int known_idx = bi * m * 3 + i * 3;
      double dist = (x1 - known[known_idx]) * (x1 - known[known_idx]) +
                    (y1 - known[known_idx + 1]) * (y1 - known[known_idx + 1]) +
                    (z1 - known[known_idx + 2]) * (z1 - known[known_idx + 2]);
      T valid_dist = dist > eps ? static_cast<T>(dist) : eps;
      if (dist < distance[input_idx]) {
        distance[input_idx + 2] = distance[input_idx + 1];
        idx[input_idx + 2] = idx[input_idx + 1];
        distance[input_idx + 1] = distance[input_idx];
        idx[input_idx + 1] = idx[input_idx];
        distance[input_idx] = dist;
        idx[input_idx] = i;
      } else if (dist < distance[input_idx + 1]) {
        distance[input_idx + 2] = distance[input_idx + 1];
        idx[input_idx + 2] = idx[input_idx + 1];
        distance[input_idx + 1] = dist;
        idx[input_idx + 1] = i;
      } else if (dist < distance[input_idx + 2]) {
        distance[input_idx + 2] = dist;
        idx[input_idx + 2] = i;
      }
    }
  }
}

template <typename T>
class ThreeNNOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* input = ctx.Input<Tensor>("X");
    auto* known = ctx.Input<Tensor>("Known");
    auto* distance = ctx.Output<Tensor>("Distance");
    auto* idx = ctx.Output<Tensor>("Idx");
    auto* input_data = input->data<T>();
    auto* known_data = known->data<T>();

    const float eps = ctx.Attr<float>("eps");

    const int b = input->dims()[0];
    const int n = input->dims()[1];
    const int m = known->dims()[1];

    auto* idx_data = idx->mutable_data<int>({b, n, 3}, ctx.GetPlace());
    auto* distance_data = distance->mutable_data<T>({b, n, 3}, ctx.GetPlace());

    int pixelNum = b * n;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeThreeNNFw<T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        distance_data, idx_data, input_data, known_data, eps, b, n, m);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(three_nn,
                        ops::ThreeNNOpCUDAKernel<float>,
                        ops::ThreeNNOpCUDAKernel<double>);
