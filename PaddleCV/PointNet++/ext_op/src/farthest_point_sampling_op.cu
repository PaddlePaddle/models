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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, unsigned int block_size>
__global__ void farthestpointsamplingKernel(int b,
                                            int n,
                                            int m,
                                            const T *__restrict__ dataset,
                                            T *__restrict__ temp,
                                            int *__restrict__ idxs) {
  // 1. add first point
  // 2. add the point having farthest distance with first point's
  // 3. make second point as first point, repeat 1,2
  if (m <= 0) return;
  const int BlockSize = block_size;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize = 3072;
  __shared__ float buf[BufferSize * 3];

  // one block one batch, n points
  // one thread one point
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    // can select old point as first point randomly
    int old = 0;
    if (threadIdx.x == 0) idxs[i * m + 0] = old;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      temp[blockIdx.x * n + j] = 1e38;
    }
    for (int j = threadIdx.x; j < min(BufferSize, n) * 3; j += blockDim.x) {
      buf[j] = dataset[i * n * 3 + j];
    }
    // wait all threads do this in the same block
    __syncthreads();

    // out m points
    for (int j = 1; j < m; j++) {
      // Step 1.
      // fatherest distance
      int besti = 0;
      float best = -1;
      // first point in m points
      float x1 = dataset[i * n * 3 + old * 3 + 0];
      float y1 = dataset[i * n * 3 + old * 3 + 1];
      float z1 = dataset[i * n * 3 + old * 3 + 2];

      // Step 2.
      // find farthest point of (x1, y1, z1)
      for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float td = temp[blockIdx.x * n + k];
        float x2, y2, z2;
        if (k < BufferSize) {
          x2 = buf[k * 3 + 0];
          y2 = buf[k * 3 + 1];
          z2 = buf[k * 3 + 2];
        } else {
          x2 = dataset[i * n * 3 + k * 3 + 0];
          y2 = dataset[i * n * 3 + k * 3 + 1];
          z2 = dataset[i * n * 3 + k * 3 + 2];
        }
        // compute eucliden distance
        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
                  (z2 - z1) * (z2 - z1);
        float d2 = min(d, td);
        if (d2 != td) temp[blockIdx.x * n + k] = d2;
        if (d2 > best) {
          best = d2;
          besti = k;
        }
      }

      // step 3.
      dists[threadIdx.x] = best;
      dists_i[threadIdx.x] = besti;
      for (int u = 0; (1 << u) < blockDim.x; u++) {
        __syncthreads();
        if (threadIdx.x < (blockDim.x >> (u + 1))) {
          int i1 = (threadIdx.x * 2) << u;
          int i2 = (threadIdx.x * 2 + 1) << u;
          if (dists[i1] < dists[i2]) {
            dists[i1] = dists[i2];
            dists_i[i1] = dists_i[i2];
          }
        }
      }
      __syncthreads();
      // store the found node index
      old = dists_i[0];
      if (threadIdx.x == 0) idxs[i * m + j] = old;
    }
  }
}

template <typename T>
class FarthestPointSamplingOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *input = ctx.Input<Tensor>("X");
    auto *output = ctx.Output<Tensor>("Output");
    if (input->numel() == 0) return;
    // allocate memory
    auto *ptr_out_points_index = output->mutable_data<int>(ctx.GetPlace());

    // b, n, m
    int batch_size = input->dims()[0];
    int in_n_points = input->dims()[1];
    int out_m_points = ctx.Attr<int>("sampled_point_num");

    const T *ptr_in_points = input->data<T>();

    Tensor tmp;
    auto *ptr_tmp_e =
        tmp.mutable_data<T>({batch_size, in_n_points}, ctx.GetPlace());

    // run fathest point sampling kernel
    // P40 have max 512 thread
    farthestpointsamplingKernel<T, 512><<<32, 512>>>(batch_size,
                                                     in_n_points,
                                                     out_m_points,
                                                     ptr_in_points,
                                                     ptr_tmp_e,
                                                     ptr_out_points_index);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(farthest_point_sampling,
                        ops::FarthestPointSamplingOpCUDAKernel<float>,
                        ops::FarthestPointSamplingOpCUDAKernel<double>);
