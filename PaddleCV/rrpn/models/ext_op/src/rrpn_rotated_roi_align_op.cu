/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Based on
@misc{ma2019rrpn,
    author = {Jianqi Ma},
    title = {{RRPN in pytorch}},
    year = {2019},
    howpublished = {\url{https://github.com/mjq11302010044/RRPN_pytorch}},
}
@article{Jianqi17RRPN,
    Author = {Jianqi Ma and Weiyuan Shao and Hao Ye and Li Wang and Hong Wang
and Yingbin Zheng and Xiangyang Xue},
    Title = {Arbitrary-Oriented Scene Text Detection via Rotation Proposals},
    journal = {IEEE Transactions on Multimedia},
    volume={20},
    number={11},
    pages={3111-3122},
    year={2018}
}*/

#include <algorithm>
#include <limits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/cuda_primitives.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;
#define PI 3.141592654

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}


template <typename T>
__global__ void Zero(T* x, int num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    x[i] = static_cast<T>(0);
  }
}

template <typename T>
__global__ void RROIAlignForward(const int nthreads,
                                 const T* bottom_data,
                                 const T spatial_scale,
                                 int height,
                                 int width,
                                 int channels,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const T* bottom_rois,
                                 int* roi_batch_id_data,
                                 T* top_data,
                                 T* con_idx_x,
                                 T* con_idx_y) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int imageWidth = width;
    int imageHeight = height;

    // (n, c, ph, pw) is an element in the pooled output
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;

    int roi_batch_ind = roi_batch_id_data[n];
    T cx = offset_bottom_rois[0];
    T cy = offset_bottom_rois[1];
    T h = offset_bottom_rois[3];
    T w = offset_bottom_rois[2];
    T angle = offset_bottom_rois[4] / 180.0 * PI;

    // TransformPrepare
    T dx = -pooled_width / 2.0;
    T dy = -pooled_height / 2.0;
    T Sx = w * spatial_scale / pooled_width;
    T Sy = h * spatial_scale / pooled_height;
    T Alpha = cos(angle);
    T Beta = sin(angle);
    T Dx = cx * spatial_scale;
    T Dy = cy * spatial_scale;

    T M[2][3];
    M[0][0] = Alpha * Sx;
    M[0][1] = Beta * Sy;
    M[0][2] = Alpha * Sx * dx + Beta * Sy * dy + Dx;
    M[1][0] = -Beta * Sx;
    M[1][1] = Alpha * Sy;
    M[1][2] = -Beta * Sx * dx + Alpha * Sy * dy + Dy;

    T P[8];
    P[0] = M[0][0] * pw + M[0][1] * ph + M[0][2];
    P[1] = M[1][0] * pw + M[1][1] * ph + M[1][2];
    P[2] = M[0][0] * pw + M[0][1] * (ph + 1) + M[0][2];
    P[3] = M[1][0] * pw + M[1][1] * (ph + 1) + M[1][2];
    P[4] = M[0][0] * (pw + 1) + M[0][1] * ph + M[0][2];
    P[5] = M[1][0] * (pw + 1) + M[1][1] * ph + M[1][2];
    P[6] = M[0][0] * (pw + 1) + M[0][1] * (ph + 1) + M[0][2];
    P[7] = M[1][0] * (pw + 1) + M[1][1] * (ph + 1) + M[1][2];

    T leftMost = (max(round(min(min(P[0], P[2]), min(P[4], P[6]))), 0.0));
    T rightMost =
        (min(round(max(max(P[0], P[2]), max(P[4], P[6]))), imageWidth - 1.0));
    T topMost = (max(round(min(min(P[1], P[3]), min(P[5], P[7]))), 0.0));
    T bottomMost =
        (min(round(max(max(P[1], P[3]), max(P[5], P[7]))), imageHeight - 1.0));

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;


    float bin_cx = (leftMost + rightMost) / 2.0;  // shift
    float bin_cy = (topMost + bottomMost) / 2.0;

    int bin_l = (int)floor(bin_cx);
    int bin_r = (int)ceil(bin_cx);
    int bin_t = (int)floor(bin_cy);
    int bin_b = (int)ceil(bin_cy);

    T lt_value = 0.0;
    if (bin_t > 0 && bin_l > 0 && bin_t < height && bin_l < width)
      lt_value = offset_bottom_data[bin_t * width + bin_l];
    T rt_value = 0.0;
    if (bin_t > 0 && bin_r > 0 && bin_t < height && bin_r < width)
      rt_value = offset_bottom_data[bin_t * width + bin_r];
    T lb_value = 0.0;
    if (bin_b > 0 && bin_l > 0 && bin_b < height && bin_l < width)
      lb_value = offset_bottom_data[bin_b * width + bin_l];
    T rb_value = 0.0;
    if (bin_b > 0 && bin_r > 0 && bin_b < height && bin_r < width)
      rb_value = offset_bottom_data[bin_b * width + bin_r];

    T rx = bin_cx - floor(bin_cx);
    T ry = bin_cy - floor(bin_cy);

    T wlt = (1.0 - rx) * (1.0 - ry);
    T wrt = rx * (1.0 - ry);
    T wrb = rx * ry;
    T wlb = (1.0 - rx) * ry;

    T inter_val = 0.0;

    inter_val += lt_value * wlt;
    inter_val += rt_value * wrt;
    inter_val += rb_value * wrb;
    inter_val += lb_value * wlb;

    platform::CudaAtomicAdd(top_data + index, static_cast<T>(inter_val));
    platform::CudaAtomicAdd(con_idx_x + index, static_cast<T>(bin_cx));
    platform::CudaAtomicAdd(con_idx_y + index, static_cast<T>(bin_cy));
  }
}

template <typename T>
__global__ void RROIAlignBackward(const int nthreads,
                                  const T* top_diff,
                                  const float* con_idx_x,
                                  const float* con_idx_y,
                                  const int num_rois,
                                  const float spatial_scale,
                                  const int height,
                                  const int width,
                                  const int channels,
                                  const int pooled_height,
                                  const int pooled_width,
                                  T* bottom_diff,
                                  const T* bottom_rois,
                                  int* roi_batch_id_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int n = index;
    n /= pooled_width;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = roi_batch_id_data[n];
    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;


    float bw = con_idx_x[index];
    float bh = con_idx_y[index];

    int bin_xs = int(floor(bw));
    int bin_ys = int(floor(bh));

    float rx = bw - float(bin_xs);
    float ry = bh - float(bin_ys);

    T wlt = (1.0 - rx) * (1.0 - ry);
    T wrt = rx * (1.0 - ry);
    T wrb = rx * ry;
    T wlb = (1.0 - rx) * ry;


    int min_x = (int)floor(bw);
    int max_x = (int)ceil(bw);
    int min_y = (int)floor(bh);
    int max_y = (int)ceil(bh);

    T top_diff_of_bin = top_diff[index];

    T v1 = wlt * top_diff_of_bin;
    T v2 = wrt * top_diff_of_bin;
    T v3 = wrb * top_diff_of_bin;
    T v4 = wlb * top_diff_of_bin;


    if (min_y > 0 && min_x > 0 && min_y < height - 1 && min_x < width - 1)
      platform::CudaAtomicAdd(offset_bottom_diff + min_y * width + min_x,
                              static_cast<T>(v1));
    if (min_y > 0 && max_x < width - 1 && min_y < height - 1 && max_x > 0)
      platform::CudaAtomicAdd(offset_bottom_diff + min_y * width + max_x,
                              static_cast<T>(v2));
    if (max_y < height - 1 && max_x < width - 1 && max_y > 0 && max_x > 0)
      platform::CudaAtomicAdd(offset_bottom_diff + max_y * width + max_x,
                              static_cast<T>(v3));
    if (max_y < height - 1 && min_x > 0 && max_y > 0 && min_x < width - 1)
      platform::CudaAtomicAdd(offset_bottom_diff + max_y * width + min_x,
                              static_cast<T>(v4));
  }
}

template <typename Place, typename T>
class RRPNROIAlignRotatedCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");
    auto* out = ctx.Output<Tensor>("Out");
    auto* con_idx_x = ctx.Output<Tensor>("ConIdX");
    auto* con_idx_y = ctx.Output<Tensor>("ConIdY");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = input->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];

    int rois_num = rois->dims()[0];

    if (rois_num == 0) return;

    int output_size = out->numel();
    int blocks = NumBlocks(output_size);
    int threads = kNumCUDAThreads;

    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto lod = rois->lod();
    PADDLE_ENFORCE_EQ(
        lod.empty(),
        false,
        "Input(ROIs) Tensor of ROIAlignOp does not contain LoD information.");
    auto rois_lod = lod.back();
    int rois_batch_size = rois_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        rois_batch_size,
        batch_size,
        "The rois_batch_size and imgs batch_size must be the same.");
    int rois_num_with_lod = rois_lod[rois_batch_size];
    PADDLE_ENFORCE_EQ(rois_num,
                      rois_num_with_lod,
                      "The rois_num from input and lod must be the same.");
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
    auto& dev_ctx = ctx.cuda_device_context();
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    auto roi_ptr = memory::Alloc(dev_ctx, bytes);
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    memory::Copy(gplace,
                 roi_id_data,
                 cplace,
                 roi_batch_id_data,
                 bytes,
                 dev_ctx.stream());

    T* out_ = out->mutable_data<T>(ctx.GetPlace());
    T* con_idx_x_ = con_idx_x->mutable_data<T>(ctx.GetPlace());
    T* con_idx_y_ = con_idx_y->mutable_data<T>(ctx.GetPlace());

    int idx_x_num = con_idx_x->numel();
    int idx_y_num = con_idx_y->numel();
    int out_num = out->numel();
    Zero<<<(idx_x_num + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(con_idx_x_,
                                                                    idx_x_num);
    Zero<<<(idx_y_num + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(con_idx_y_,
                                                                    idx_y_num);
    Zero<<<(out_num + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(out_,
                                                                  out_num);

    RROIAlignForward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        output_size,
        input->data<T>(),
        spatial_scale,
        height,
        width,
        channels,
        pooled_height,
        pooled_width,
        rois->data<T>(),
        roi_id_data,
        out_,
        con_idx_x_,
        con_idx_y_);
  }
};

template <typename Place, typename T>
class RRPNROIAlignRotatedGradCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* rois = ctx.Input<LoDTensor>("ROIs");

    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* con_idx_x = ctx.Input<Tensor>("ConIdX");
    auto* con_idx_y = ctx.Input<Tensor>("ConIdY");
    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    int rois_num = rois->dims()[0];
    int channels = input->dims()[1];
    int height = input->dims()[2];
    int width = input->dims()[3];

    if (!in_grad) {
      return;
    }
    Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto rois_lod = rois->lod().back();
    int rois_batch_size = rois_lod.size() - 1;
    for (int n = 0; n < rois_batch_size; ++n) {
      for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
    auto& dev_ctx = ctx.cuda_device_context();
    auto roi_ptr =
        memory::Alloc(dev_ctx, roi_batch_id_list.numel() * sizeof(int));
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    memory::Copy(gplace,
                 roi_id_data,
                 cplace,
                 roi_batch_id_data,
                 bytes,
                 dev_ctx.stream());
    T* in_grad_ = in_grad->mutable_data<T>(ctx.GetPlace());
    int in_grad_num = in_grad->numel();
    Zero<<<(in_grad_num + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
        in_grad_, in_grad_num);
    int output_grad_size = out_grad->numel();
    int blocks = NumBlocks(output_grad_size);
    int threads = kNumCUDAThreads;
    con_idx_x->data<float>();
    con_idx_y->data<float>();
    out_grad->data<T>();
    rois->data<T>();
    if (output_grad_size > 0) {
      RROIAlignBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
          output_grad_size,
          out_grad->data<T>(),
          con_idx_x->data<float>(),
          con_idx_y->data<float>(),
          rois_num,
          spatial_scale,
          height,
          width,
          channels,
          pooled_height,
          pooled_width,
          in_grad_,
          // in_grad->mutable_data<T>(ctx.GetPlace()),
          rois->data<T>(),
          roi_id_data);
    }
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    rrpn_rotated_roi_align,
    ops::RRPNROIAlignRotatedCUDAKernel<paddle::platform::CUDADeviceContext,
                                       float>,
    ops::RRPNROIAlignRotatedCUDAKernel<paddle::platform::CUDADeviceContext,
                                       double>);
REGISTER_OP_CUDA_KERNEL(
    rrpn_rotated_roi_align_grad,
    ops::RRPNROIAlignRotatedGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                           float>,
    ops::RRPNROIAlignRotatedGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                           double>);
