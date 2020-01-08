/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
--------------------------------------------------------
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
}
--------------------------------------------------------
*/

#include <paddle/fluid/memory/allocation/allocator.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "cub/cub/cub.cuh"
#include "gather.cu.h"
#include "math_function.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/for_range.h"
#include "safe_ref.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
#define PI 3.141592654

namespace {

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

int const kThreadsPerBlock = sizeof(uint64_t) * 8;

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

struct RangeInitFunctor {
  int start_;
  int delta_;
  int *out_;
  __device__ void operator()(size_t i) { out_[i] = start_ + i * delta_; }
};

template <typename T>
static void RSortDescending(const platform::CUDADeviceContext &ctx,
                            const Tensor &value,
                            Tensor *value_out,
                            Tensor *index_out) {
  int num = static_cast<int>(value.numel());
  Tensor index_in_t;
  int *idx_in = index_in_t.mutable_data<int>({num}, ctx.GetPlace());
  platform::ForRange<platform::CUDADeviceContext> for_range(ctx, num);
  for_range(RangeInitFunctor{0, 1, idx_in});

  int *idx_out = index_out->mutable_data<int>({num}, ctx.GetPlace());

  const T *keys_in = value.data<T>();
  T *keys_out = value_out->mutable_data<T>({num}, ctx.GetPlace());

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      nullptr, temp_storage_bytes, keys_in, keys_out, idx_in, idx_out, num);
  // Allocate temporary storage
  auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending<T, int>(d_temp_storage->ptr(),
                                                    temp_storage_bytes,
                                                    keys_in,
                                                    keys_out,
                                                    idx_in,
                                                    idx_out,
                                                    num);
}

template <typename T>
struct RBoxDecodeAndClipFunctor {
  const T *anchor;
  const T *deltas;
  const T *var;
  const int *index;
  const T *im_info;

  T *proposals;

  RBoxDecodeAndClipFunctor(const T *anchor,
                           const T *deltas,
                           const T *var,
                           const int *index,
                           const T *im_info,
                           T *proposals)
      : anchor(anchor),
        deltas(deltas),
        var(var),
        index(index),
        im_info(im_info),
        proposals(proposals) {}

  T bbox_clip_default{static_cast<T>(kBBoxClipDefault)};

  __device__ void operator()(size_t i) {
    int k = index[i] * 5;

    T w = anchor[k + 2];
    T h = anchor[k + 3];
    T cx = anchor[k];
    T cy = anchor[k + 1];
    T angle = anchor[k + 4];

    T de_cx = deltas[k];
    T de_cy = deltas[k + 1];
    T de_w = deltas[k + 2];
    T de_h = deltas[k + 3];
    T de_g = deltas[k + 4];

    T d_cx, d_cy, d_w, d_h, d_g;
    if (var) {
      d_cx = cx + de_cx * w / var[k];
      d_cy = cy + de_cy * h / var[k + 1];
      d_w = exp(Min(de_w / var[k + 2], bbox_clip_default)) * w;
      d_h = exp(Min(de_h / var[k + 3], bbox_clip_default)) * h;
      d_g = de_g / var[k + 4] * 1.0 / PI * 180 + angle;
    } else {
      d_cx = cx + de_cx * w;
      d_cy = cy + de_cy * h;
      d_w = exp(Min(de_w, bbox_clip_default)) * w;
      d_h = exp(Min(de_h, bbox_clip_default)) * h;
      d_g = de_g * 1.0 / PI * 180 + angle;
    }

    proposals[i * 5] = d_cx;
    proposals[i * 5 + 1] = d_cy;
    proposals[i * 5 + 2] = d_w;
    proposals[i * 5 + 3] = d_h;
    proposals[i * 5 + 4] = d_g;
  }

  __device__ __forceinline__ T Min(T a, T b) const { return a > b ? b : a; }

  __device__ __forceinline__ T Max(T a, T b) const { return a > b ? a : b; }
};

template <typename T, int BlockSize>
static __global__ void RFilterBBoxes(const T *bboxes,
                                     const T *im_info,
                                     const T min_size,
                                     const int num,
                                     int *keep_num,
                                     int *keep) {
  T im_h = im_info[0];
  T im_w = im_info[1];
  T im_scale = im_info[2];

  int cnt = 0;
  __shared__ int keep_index[BlockSize];

  CUDA_1D_KERNEL_LOOP(i, num) {
    keep_index[threadIdx.x] = -1;
    __syncthreads();

    int k = i * 5;

    T cx = bboxes[k];
    T cy = bboxes[k + 1];
    T w_s = bboxes[k + 2];
    T h_s = bboxes[k + 3];

    if (w_s >= min_size && h_s >= min_size) {
      keep_index[threadIdx.x] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int size = (num - i) < BlockSize ? num - i : BlockSize;
      for (int j = 0; j < size; ++j) {
        if (keep_index[j] > -1) {
          keep[cnt++] = keep_index[j];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    keep_num[0] = cnt;
  }
}


__device__ inline float trangle_area(float *a, float *b, float *c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0;
}


__device__ inline float area(float *int_pts, int num_of_inter) {
  float area = 0.0;
  for (int i = 0; i < num_of_inter - 2; i++) {
    area +=
        fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}


__device__ inline void reorder_pts(float *int_pts, int num_of_inter) {
  if (num_of_inter > 0) {
    float center[2] = {0.0, 0.0};

    //    center[0] = 0.0;
    //    center[1] = 0.0;

    for (int i = 0; i < num_of_inter; i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for (int i = 0; i < num_of_inter; i++) {
      v[0] = int_pts[2 * i] - center[0];
      v[1] = int_pts[2 * i + 1] - center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if (v[1] < 0) {
        v[0] = -2 - v[0];
      }
      vs[i] = v[0];
    }

    float temp, tx, ty;
    int j;
    for (int i = 1; i < num_of_inter; ++i) {
      if (vs[i - 1] > vs[i]) {
        temp = vs[i];
        tx = int_pts[2 * i];
        ty = int_pts[2 * i + 1];
        j = i;
        while (j > 0 && vs[j - 1] > temp) {
          vs[j] = vs[j - 1];
          int_pts[j * 2] = int_pts[j * 2 - 2];
          int_pts[j * 2 + 1] = int_pts[j * 2 - 1];
          j--;
        }
        vs[j] = temp;
        int_pts[j * 2] = tx;
        int_pts[j * 2 + 1] = ty;
      }
    }
  }
}


__device__ inline bool inter2line(
    float *pts1, float *pts2, int i, int j, float *temp_pts) {
  float a[2] = {pts1[2 * i], pts1[2 * i + 1]};
  float b[2] = {pts1[2 * ((i + 1) % 4)], pts1[2 * ((i + 1) % 4) + 1]};
  float c[2] = {pts2[2 * j], pts2[2 * j + 1]};
  float d[2] = {pts2[2 * ((j + 1) % 4)], pts2[2 * ((j + 1) % 4) + 1]};

  // T area_abc, area_abd, area_cda, area_cdb;

  // a[0] = pts1[2 * i];
  // a[1] = pts1[2 * i + 1];

  // b[0] = pts1[2 * ((i + 1) % 4)];
  // b[1] = pts1[2 * ((i + 1) % 4) + 1];

  // c[0] = pts2[2 * j];
  // c[1] = pts2[2 * j + 1];

  // d[0] = pts2[2 * ((j + 1) % 4)];
  // d[1] = pts2[2 * ((j + 1) % 4) + 1];

  float area_abc = trangle_area(a, b, c);
  float area_abd = trangle_area(a, b, d);

  if (area_abc * area_abd >= 0) {
    return false;
  }

  float area_cda = trangle_area(c, d, a);
  float area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= 0) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);

  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}


__device__ inline bool in_rect(float pt_x, float pt_y, float *pts) {
  float ab[2] = {pts[2] - pts[0], pts[3] - pts[1]};
  float ad[2] = {pts[6] - pts[0], pts[7] - pts[1]};
  float ap[2] = {pt_x - pts[0], pt_y - pts[1]};

  //  float abab;
  //  float abap;
  //  float adad;
  //  float adap;

  //  ab[0] = pts[2] - pts[0];
  //  ab[1] = pts[3] - pts[1];
  //
  //  ad[0] = pts[6] - pts[0];
  //  ad[1] = pts[7] - pts[1];
  //
  //  ap[0] = pt_x - pts[0];
  //  ap[1] = pt_y - pts[1];

  float abab = ab[0] * ab[0] + ab[1] * ab[1];
  float abap = ab[0] * ap[0] + ab[1] * ap[1];
  float adad = ad[0] * ad[0] + ad[1] * ad[1];
  float adap = ad[0] * ap[0] + ad[1] * ap[1];

  return abab >= abap and abap >= 0 and adad >= adap and adap >= 0;
}


__device__ inline int inter_pts(float *pts1, float *pts2, float *int_pts) {
  int num_of_inter = 0;

  for (int i = 0; i < 4; i++) {
    if (in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
    if (in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  float temp_pts[2];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if (has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }

  return num_of_inter;
}


__device__ inline void convert_region(float *pts, const float *region) {
  float angle = region[4];
  float a_cos = cos(angle / 180.0 * PI);
  float a_sin = -sin(angle / 180.0 * PI);  // anti clock-wise

  float ctr_x = region[0];
  float ctr_y = region[1];
  float h = region[3];
  float w = region[2];

  float pts_x[4] = {-w / 2, -w / 2, w / 2, w / 2};
  float pts_y[4] = {-h / 2, h / 2, h / 2, -h / 2};

  //  pts_x[0] = -w / 2;
  //  pts_x[1] = -w / 2;
  //  pts_x[2] = w / 2;
  //  pts_x[3] = w / 2;
  //
  //  pts_y[0] = -h / 2;
  //  pts_y[1] = h / 2;
  //  pts_y[2] = h / 2;
  //  pts_y[3] = -h / 2;

  for (int i = 0; i < 4; i++) {
    pts[2 * i] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    pts[2 * i + 1] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;
  }
}

__device__ inline float inter(const float *region1, const float *region2) {
  float pts1[8];
  float pts2[8];
  float int_pts[16];
  int num_of_inter;

  convert_region(pts1, region1);
  convert_region(pts2, region2);

  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);
}


__device__ inline float IoU(const float *region1, const float *region2) {
  float area1 = region1[2] * region1[3];
  float area2 = region2[2] * region2[3];
  float area_inter = inter(region1, region2);

  return area_inter / (area1 + area2 - area_inter);
}

static __global__ void RNMSKernel(const int n_boxes,
                                  const float nms_overlap_thresh,
                                  const float *dev_boxes,
                                  uint64_t *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_boxes - row_start * kThreadsPerBlock, kThreadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * kThreadsPerBlock, kThreadsPerBlock);

  __shared__ float block_boxes[kThreadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = kThreadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (IoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, kThreadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template <typename T>
static void RNMS(const platform::CUDADeviceContext &ctx,
                 const Tensor &proposals,
                 const Tensor &sorted_indices,
                 const T nms_threshold,
                 Tensor *keep_out) {
  int boxes_num = proposals.dims()[0];
  PADDLE_ENFORCE_EQ(boxes_num, sorted_indices.dims()[0]);

  const int col_blocks = DIVUP(boxes_num, kThreadsPerBlock);
  dim3 blocks(DIVUP(boxes_num, kThreadsPerBlock),
              DIVUP(boxes_num, kThreadsPerBlock));
  dim3 threads(kThreadsPerBlock);

  const T *boxes = proposals.data<T>();
  auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  framework::Vector<uint64_t> mask(boxes_num * col_blocks);
  RNMSKernel<<<blocks, threads>>>(
      boxes_num,
      nms_threshold,
      boxes,
      mask.CUDAMutableData(boost::get<platform::CUDAPlace>(ctx.GetPlace())));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  std::vector<int> keep_vec;
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / kThreadsPerBlock;
    int inblock = i % kThreadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      ++num_to_keep;
      keep_vec.push_back(i);
      uint64_t *p = &mask[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  int *keep = keep_out->mutable_data<int>({num_to_keep}, ctx.GetPlace());
  memory::Copy(place,
               keep,
               platform::CPUPlace(),
               keep_vec.data(),
               sizeof(int) * num_to_keep,
               ctx.stream());
  ctx.Wait();
}

template <typename T>
static std::pair<Tensor, Tensor> RRPNProposalForOneImage(
    const platform::CUDADeviceContext &ctx,
    const Tensor &im_info,
    const Tensor &anchors,
    const Tensor &variances,
    const Tensor &bbox_deltas,  // [M, 5]
    const Tensor &scores,       // [N, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size) {
  // 1. pre nms
  Tensor scores_sort, index_sort;
  RSortDescending<T>(ctx, scores, &scores_sort, &index_sort);
  int num = scores.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num) ? scores.numel()
                                                                : pre_nms_top_n;
  scores_sort.Resize({pre_nms_num, 1});
  index_sort.Resize({pre_nms_num, 1});

  // 2. box decode and clipping
  Tensor proposals;
  proposals.mutable_data<T>({pre_nms_num, 5}, ctx.GetPlace());

  {
    platform::ForRange<platform::CUDADeviceContext> for_range(ctx, pre_nms_num);
    for_range(RBoxDecodeAndClipFunctor<T>{anchors.data<T>(),
                                          bbox_deltas.data<T>(),
                                          variances.data<T>(),
                                          index_sort.data<int>(),
                                          im_info.data<T>(),
                                          proposals.data<T>()});
  }

  // 3. filter
  Tensor keep_index, keep_num_t;
  keep_index.mutable_data<int>({pre_nms_num}, ctx.GetPlace());
  keep_num_t.mutable_data<int>({1}, ctx.GetPlace());
  min_size = std::max(min_size, 0.0f);
  auto stream = ctx.stream();
  RFilterBBoxes<T, 256><<<1, 256, 0, stream>>>(proposals.data<T>(),
                                               im_info.data<T>(),
                                               min_size,
                                               pre_nms_num,
                                               keep_num_t.data<int>(),
                                               keep_index.data<int>());
  int keep_num;
  const auto gpu_place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  memory::Copy(platform::CPUPlace(),
               &keep_num,
               gpu_place,
               keep_num_t.data<int>(),
               sizeof(int),
               ctx.stream());
  ctx.Wait();
  keep_index.Resize({keep_num});

  Tensor scores_filter, proposals_filter;
  proposals_filter.mutable_data<T>({keep_num, 5}, ctx.GetPlace());
  scores_filter.mutable_data<T>({keep_num, 1}, ctx.GetPlace());
  GPUGather<T>(ctx, proposals, keep_index, &proposals_filter);
  GPUGather<T>(ctx, scores_sort, keep_index, &scores_filter);

  if (nms_thresh <= 0) {
    return std::make_pair(proposals_filter, scores_filter);
  }

  // 4. nms
  Tensor keep_nms;
  RNMS<T>(ctx, proposals_filter, keep_index, nms_thresh, &keep_nms);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize({post_nms_top_n});
  }

  Tensor scores_nms, proposals_nms;
  proposals_nms.mutable_data<T>({keep_nms.numel(), 5}, ctx.GetPlace());
  scores_nms.mutable_data<T>({keep_nms.numel(), 1}, ctx.GetPlace());
  GPUGather<T>(ctx, proposals_filter, keep_nms, &proposals_nms);
  GPUGather<T>(ctx, scores_filter, keep_nms, &scores_nms);

  return std::make_pair(proposals_nms, scores_nms);
}
}  // namespace

template <typename DeviceContext, typename T>
class CUDARRPNGenerateProposalsKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<Tensor>("Scores");
    auto *bbox_deltas = context.Input<Tensor>("BboxDeltas");
    auto *im_info = context.Input<Tensor>("ImInfo");
    auto anchors = detail::Ref(context.Input<Tensor>("Anchors"),
                               "Cannot find input Anchors(%s) in scope",
                               context.InputNames("Anchors")[0]);
    auto variances = detail::Ref(context.Input<Tensor>("Variances"),
                                 "Cannot find input Variances(%s) in scope",
                                 context.InputNames("Variances")[0]);

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");

    int pre_nms_top_n = context.Attr<int>("pre_nms_topN");
    int post_nms_top_n = context.Attr<int>("post_nms_topN");
    float nms_thresh = context.Attr<float>("nms_thresh");
    float min_size = context.Attr<float>("min_size");

    auto &dev_ctx = context.template device_context<DeviceContext>();

    auto scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    math::Transpose<DeviceContext, T, 4> trans;
    std::vector<int> axis = {0, 2, 3, 1};
    trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
    trans(dev_ctx, *scores, &scores_swap, axis);

    anchors.Resize({anchors.numel() / 5, 5});
    variances.Resize({variances.numel() / 5, 5});

    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 5, 5},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    T *rpn_rois_data = rpn_rois->data<T>();
    T *rpn_roi_probs_data = rpn_roi_probs->data<T>();

    auto place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());

    int64_t num_proposals = 0;
    std::vector<size_t> offset(1, 0);
    for (int64_t i = 0; i < num; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 5, 5});
      scores_slice.Resize({h_score * w_score * c_score, 1});
      // auto* scores_data = scores_slice.data<T>();
      // for(int k=0; k < 256; k++) {
      //     std::cout << scores_data[k] << std::endl;
      // }
      std::pair<Tensor, Tensor> box_score_pair =
          RRPNProposalForOneImage<T>(dev_ctx,
                                     im_info_slice,
                                     anchors,
                                     variances,
                                     bbox_deltas_slice,
                                     scores_slice,
                                     pre_nms_top_n,
                                     post_nms_top_n,
                                     nms_thresh,
                                     min_size);

      Tensor &proposals = box_score_pair.first;
      Tensor &scores = box_score_pair.second;

      memory::Copy(place,
                   rpn_rois_data + num_proposals * 5,
                   place,
                   proposals.data<T>(),
                   sizeof(T) * proposals.numel(),
                   dev_ctx.stream());
      memory::Copy(place,
                   rpn_roi_probs_data + num_proposals,
                   place,
                   scores.data<T>(),
                   sizeof(T) * scores.numel(),
                   dev_ctx.stream());
      dev_ctx.Wait();
      num_proposals += proposals.dims()[0];
      offset.emplace_back(num_proposals);
    }
    framework::LoD lod;
    lod.emplace_back(offset);
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 5});
    rpn_roi_probs->Resize({num_proposals, 1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    rrpn_generate_proposals,
    ops::CUDARRPNGenerateProposalsKernel<paddle::platform::CUDADeviceContext,
                                         float>);
