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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <vector>
#include "paddle/fluid/memory/memory.h"
//#include "rrpn_box_coder_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

#define PI 3.141592654

template <typename T>
__global__ void DecodeCenterSizeKernel(const T* prior_box_data,
                                       const T* prior_box_var_data,
                                       const T* target_box_data,
                                       const int row,
                                       const int len,
                                       const T prior_box_var_size,
                                       const float* variance,
                                       const int var_size,
                                       T* output) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int prior_box_offset = 0;
  if (idx < row) {
    const int row_idx = idx;
    prior_box_offset = row_idx * len;
    T prior_box_width = prior_box_data[prior_box_offset + 2];

    T prior_box_height = prior_box_data[prior_box_offset + 3];

    T prior_box_center_x = prior_box_data[prior_box_offset];
    T prior_box_center_y = prior_box_data[prior_box_offset + 1];
    T prior_box_angle = prior_box_data[prior_box_offset + 4];

    T target_box_width, target_box_height, target_box_angle;
    T target_box_center_x, target_box_center_y;
    T box_var_x = T(1), box_var_y = T(1);
    T box_var_w = T(1), box_var_h = T(1), box_var_angle = T(1);
    if (prior_box_var_data) {
      int prior_var_offset = row_idx * len;
      box_var_x = prior_box_var_data[prior_var_offset];
      box_var_y = prior_box_var_data[prior_var_offset + 1];
      box_var_w = prior_box_var_data[prior_var_offset + 2];
      box_var_h = prior_box_var_data[prior_var_offset + 3];
      box_var_angle = prior_box_var_data[prior_var_offset + 4];
    } else if (var_size == 5) {
      box_var_x = static_cast<T>(variance[0]);
      box_var_y = static_cast<T>(variance[1]);
      box_var_w = static_cast<T>(variance[2]);
      box_var_h = static_cast<T>(variance[3]);
      box_var_angle = static_cast<T>(variance[4]);
    }
    target_box_width =
        exp(target_box_data[idx * len + 2] / box_var_w) * prior_box_width / 1.4;
    target_box_height = exp(target_box_data[idx * len + 3] / box_var_h) *
                        prior_box_height / 1.4;
    target_box_center_x =
        target_box_data[idx * len] / box_var_x * prior_box_width +
        prior_box_center_x;
    target_box_center_y =
        target_box_data[idx * len + 1] / box_var_y * prior_box_height +
        prior_box_center_y;

    target_box_angle =
        (target_box_data[idx * len + 4] / box_var_angle) * 1.0 / PI * 180 +
        prior_box_angle;

    T a_cos = cos(PI / 180 * target_box_angle);
    T a_sin = -sin(PI / 180 * target_box_angle);

    T rotation_matrix[3][3];

    rotation_matrix[0][0] = a_cos;
    rotation_matrix[0][1] = a_sin;
    rotation_matrix[0][2] = 0;
    rotation_matrix[1][0] = -a_sin;
    rotation_matrix[1][1] = a_cos;
    rotation_matrix[1][2] = 0;
    rotation_matrix[2][0] = -target_box_center_x * a_cos +
                            target_box_center_y * a_sin + target_box_center_x;
    rotation_matrix[2][1] = -target_box_center_x * a_sin -
                            target_box_center_y * a_cos + target_box_center_y;
    rotation_matrix[2][2] = 1;

    T pt_x0 = target_box_center_x - target_box_width / 2;
    T pt_x1 = target_box_center_x + target_box_width / 2;
    T pt_x2 = target_box_center_x + target_box_width / 2;
    T pt_x3 = target_box_center_x - target_box_width / 2;

    T pt_y0 = target_box_center_y - target_box_height / 2;
    T pt_y1 = target_box_center_y - target_box_height / 2;
    T pt_y2 = target_box_center_y + target_box_height / 2;
    T pt_y3 = target_box_center_y + target_box_height / 2;


    output[idx * 8] = pt_x0 * rotation_matrix[0][0] +
                      pt_y0 * rotation_matrix[1][0] + rotation_matrix[2][0];
    output[idx * 8 + 1] = pt_x0 * rotation_matrix[0][1] +
                          pt_y0 * rotation_matrix[1][1] + rotation_matrix[2][1];
    output[idx * 8 + 2] = pt_x1 * rotation_matrix[0][0] +
                          pt_y1 * rotation_matrix[1][0] + rotation_matrix[2][0];
    output[idx * 8 + 3] = pt_x1 * rotation_matrix[0][1] +
                          pt_y1 * rotation_matrix[1][1] + rotation_matrix[2][1];
    output[idx * 8 + 4] = pt_x2 * rotation_matrix[0][0] +
                          pt_y2 * rotation_matrix[1][0] + rotation_matrix[2][0];
    output[idx * 8 + 5] = pt_x2 * rotation_matrix[0][1] +
                          pt_y2 * rotation_matrix[1][1] + rotation_matrix[2][1];
    output[idx * 8 + 6] = pt_x3 * rotation_matrix[0][0] +
                          pt_y3 * rotation_matrix[1][0] + rotation_matrix[2][0];
    output[idx * 8 + 7] = pt_x3 * rotation_matrix[0][1] +
                          pt_y3 * rotation_matrix[1][1] + rotation_matrix[2][1];
  }
}

template <typename DeviceContext, typename T>
class RRPNBoxCoderCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* prior_box = context.Input<framework::Tensor>("PriorBox");
    auto* prior_box_var = context.Input<framework::Tensor>("PriorBoxVar");
    auto* target_box = context.Input<framework::LoDTensor>("TargetBox");
    auto* output_box = context.Output<framework::Tensor>("OutputBox");
    std::vector<float> variance = context.Attr<std::vector<float>>("variance");
    const T* prior_box_data = prior_box->data<T>();
    const T* target_box_data = target_box->data<T>();
    const T* prior_box_var_data = nullptr;
    auto prior_box_var_size = 0;
    if (prior_box_var) {
      PADDLE_ENFORCE(variance.empty(),
                     "Input 'PriorBoxVar' and attribute 'variance' should not"
                     "be used at the same time.");
      prior_box_var_data = prior_box_var->data<T>();
      prior_box_var_size = prior_box_var->dims().size();
    }
    if (!(variance.empty())) {
      PADDLE_ENFORCE(static_cast<int>(variance.size()) == 5,
                     "Size of attribute 'variance' should be 4");
    }

    if (target_box->lod().size()) {
      PADDLE_ENFORCE_EQ(
          target_box->lod().size(), 1, "Only support 1 level of LoD.");
    }
    const int var_size = static_cast<int>(variance.size());
    auto row = target_box->dims()[0];
    auto len = 5;
    int block = 512;
    int grid = (row + block - 1) / block;
    auto& device_ctx = context.cuda_device_context();

    int bytes = var_size * sizeof(float);
    auto dev_var = memory::Alloc(device_ctx, bytes);
    float* dev_var_data = reinterpret_cast<float*>(dev_var->ptr());
    auto cplace = platform::CPUPlace();
    const auto gplace = boost::get<platform::CUDAPlace>(context.GetPlace());
    memory::Copy(
        gplace, dev_var_data, cplace, &variance[0], bytes, device_ctx.stream());

    output_box->mutable_data<T>({row, 8}, context.GetPlace());
    T* output = output_box->data<T>();

    DecodeCenterSizeKernel<T><<<grid, block, 0, device_ctx.stream()>>>(
        prior_box_data,
        prior_box_var_data,
        target_box_data,
        row,
        len,
        prior_box_var_size,
        dev_var_data,
        var_size,
        output);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    rrpn_box_coder,
    ops::RRPNBoxCoderCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RRPNBoxCoderCUDAKernel<paddle::platform::CUDADeviceContext, double>);
