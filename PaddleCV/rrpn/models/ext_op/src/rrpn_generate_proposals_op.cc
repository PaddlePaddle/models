/*opyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "gather.h"
#include "math_function.h"
#include "paddle/fluid/framework/op_registry.h"
#include "safe_ref.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);
#define PI 3.141592654

static void RRPNAppendProposals(Tensor *dst,
                                int64_t offset,
                                const Tensor &src) {
  auto *out_data = dst->data<void>();
  auto *to_add_data = src.data<void>();
  size_t size_of_t = framework::SizeOfType(src.type());
  offset *= size_of_t;
  std::memcpy(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(out_data) + offset),
      to_add_data,
      src.numel() * size_of_t);
}

template <class T>
inline T axr(T x, T r) {
  return 0.5 * PI * r * r - x * sqrt(r * r - x * x) - r * r * std::asin(x / r);
}

class RRPNGenerateProposalsOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Scores"), "Input(Scores) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("BboxDeltas"),
                   "Input(BboxDeltas) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("ImInfo"), "Input(ImInfo) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Anchors"),
                   "Input(Anchors) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Variances"),
                   "Input(Variances) shouldn't be null.");

    ctx->SetOutputDim("RpnRois", {-1, 5});
    ctx->SetOutputDim("RpnRoiProbs", {-1, 1});
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Anchors")->type(),
                                   ctx.device_context());
  }
};

template <class T>
static inline void RBoxCoder(const platform::DeviceContext &ctx,
                             Tensor *all_anchors,
                             Tensor *bbox_deltas,
                             Tensor *variances,
                             Tensor *proposals) {
  T *proposals_data = proposals->mutable_data<T>(ctx.GetPlace());

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto *bbox_deltas_data = bbox_deltas->data<T>();
  auto *anchor_data = all_anchors->data<T>();
  const T *variances_data = nullptr;
  if (variances) {
    variances_data = variances->data<T>();
  }

  for (int64_t i = 0; i < row; ++i) {
    T anchor_width = anchor_data[i * len + 2];
    T anchor_height = anchor_data[i * len + 3];
    T anchor_angle = anchor_data[i * len + 4];

    T anchor_center_x = anchor_data[i * len];
    T anchor_center_y = anchor_data[i * len + 1];

    T bbox_center_x = 0, bbox_center_y = 0;
    T bbox_width = 0, bbox_height = 0, bbox_angle = 0;

    if (variances) {
      bbox_center_x =
          bbox_deltas_data[i * len] / variances_data[i * len] * anchor_width +
          anchor_center_x;
      bbox_center_y = bbox_deltas_data[i * len + 1] /
                          variances_data[i * len + 1] * anchor_height +
                      anchor_center_y;
      bbox_width = std::exp(std::min<T>(bbox_deltas_data[i * len + 2] /
                                            variances_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(bbox_deltas_data[i * len + 3] /
                                             variances_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
      bbox_angle =
          (bbox_deltas_data[i * len + 4] / variances_data[i * len + 4]) * 1.0 /
              PI * 180 +
          anchor_angle;

    } else {
      bbox_center_x =
          bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
      bbox_center_y =
          bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
      bbox_width = std::exp(std::min<T>(bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
      bbox_angle =
          bbox_deltas_data[i * len + 4] * 1.0 / PI * 180 + anchor_angle;
    }

    proposals_data[i * len] = bbox_center_x;
    proposals_data[i * len + 1] = bbox_center_y;
    proposals_data[i * len + 2] = bbox_width;
    proposals_data[i * len + 3] = bbox_height;
    proposals_data[i * len + 4] = bbox_angle;
  }
  // return proposals;
}


template <class T>
static inline void RFilterBoxes(const platform::DeviceContext &ctx,
                                Tensor *boxes,
                                float min_size,
                                const Tensor &im_info,
                                Tensor *keep) {
  T *boxes_data = boxes->mutable_data<T>(ctx.GetPlace());
  keep->Resize({boxes->dims()[0]});
  min_size = std::max(min_size, 0.0f);
  int *keep_data = keep->mutable_data<int>(ctx.GetPlace());

  int keep_len = 0;
  for (int i = 0; i < boxes->dims()[0]; ++i) {
    T ws = boxes_data[5 * i + 2];
    T hs = boxes_data[5 * i + 3];
    if (ws >= min_size && hs >= min_size) {
      keep_data[keep_len++] = i;
    }
  }
  keep->Resize({keep_len});
}

template <class T>
static inline std::vector<std::pair<T, int>> GetSortedScoreIndex(
    const std::vector<T> &scores) {
  std::vector<std::pair<T, int>> sorted_indices;
  sorted_indices.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    sorted_indices.emplace_back(scores[i], i);
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices.begin(),
                   sorted_indices.end(),
                   [](const std::pair<T, int> &a, const std::pair<T, int> &b) {
                     return a.first < b.first;
                   });
  return sorted_indices;
}


template <typename T>
static inline Tensor VectorToTensor(const std::vector<T> &selected_indices,
                                    int selected_num) {
  Tensor keep_nms;
  keep_nms.Resize({selected_num});
  auto *keep_data = keep_nms.mutable_data<T>(platform::CPUPlace());
  for (int i = 0; i < selected_num; ++i) {
    keep_data[i] = selected_indices[i];
  }
  return keep_nms;
}

template <typename T>
inline T trangle_area(T *a, T *b, T *c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0;
}

template <typename T>
inline T area(T *int_pts, int num_of_inter) {
  float area = 0.0;
  for (int i = 0; i < num_of_inter - 2; i++) {
    area +=
        fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}

template <typename T>
inline void reorder_pts(T *int_pts, int num_of_inter) {
  if (num_of_inter > 0) {
    float center[2];

    center[0] = 0.0;
    center[1] = 0.0;

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

template <typename T>
inline bool inter2line(T *pts1, T *pts2, int i, int j, T *temp_pts) {
  T a[2];
  T b[2];
  T c[2];
  T d[2];

  T area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);

  if (area_abc * area_abd >= 0) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

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

template <typename T>
inline bool in_rect(T pt_x, T pt_y, T *pts) {
  float ab[2];
  float ad[2];
  float ap[2];

  float abab;
  float abap;
  float adad;
  float adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];

  return abab >= abap and abap >= 0 and adad >= adap and adap >= 0;
}

template <typename T>
inline int inter_pts(T *pts1, T *pts2, T *int_pts) {
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

  T temp_pts[2];

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

template <typename T>
inline void convert_region(T *pts, const T *region) {
  float angle = region[4];
  float a_cos = cos(angle / 180.0 * PI);
  float a_sin = -sin(angle / 180.0 * PI);  // anti clock-wise

  float ctr_x = region[0];
  float ctr_y = region[1];
  float h = region[3];
  float w = region[2];

  float pts_x[4];
  float pts_y[4];

  pts_x[0] = -w / 2;
  pts_x[1] = -w / 2;
  pts_x[2] = w / 2;
  pts_x[3] = w / 2;

  pts_y[0] = -h / 2;
  pts_y[1] = h / 2;
  pts_y[2] = h / 2;
  pts_y[3] = -h / 2;

  for (int i = 0; i < 4; i++) {
    pts[2 * i] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    pts[2 * i + 1] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;
  }
}

template <typename T>
inline float inter(const T *region1, const T *region2) {
  T pts1[8];
  T pts2[8];
  T int_pts[16];
  int num_of_inter;

  convert_region<T>(pts1, region1);
  convert_region<T>(pts2, region2);

  num_of_inter = inter_pts<T>(pts1, pts2, int_pts);

  reorder_pts<T>(int_pts, num_of_inter);

  return area<T>(int_pts, num_of_inter);
}

template <typename T>
inline float DevRotateIoU(const T *region1, const T *region2) {
  T area1 = region1[2] * region1[3];
  T area2 = region2[2] * region2[3];
  T area_inter = inter<T>(region1, region2);

  return area_inter / (area1 + area2 - area_inter);
}

template <class T>
static inline Tensor RNMS(const platform::DeviceContext &ctx,
                          Tensor *bbox,
                          Tensor *scores,
                          T nms_threshold) {
  PADDLE_ENFORCE_NOT_NULL(bbox);
  int64_t num_boxes = bbox->dims()[0];
  // 4: [xmin ymin xmax ymax]
  int64_t box_size = bbox->dims()[1];

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores->data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices =
      GetSortedScoreIndex<T>(scores_data);

  std::vector<int> selected_indices;
  int selected_num = 0;
  T adaptive_threshold = nms_threshold;
  const T *bbox_data = bbox->data<T>();
  while (sorted_indices.size() != 0) {
    int idx = sorted_indices.back().second;
    bool flag = true;
    for (int kept_idx : selected_indices) {
      if (flag) {
        T overlap = DevRotateIoU<T>(bbox_data + idx * box_size,
                                    bbox_data + kept_idx * box_size);
        flag = (overlap <= adaptive_threshold);
      } else {
        break;
      }
    }
    if (flag) {
      selected_indices.push_back(idx);
      ++selected_num;
    }
    sorted_indices.erase(sorted_indices.end() - 1);
  }
  return VectorToTensor(selected_indices, selected_num);
}

template <typename T>
class RRPNGenerateProposalsKernel : public framework::OpKernel<T> {
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

    auto &dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();

    auto &scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto &bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 5, 5},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    math::Transpose<platform::CPUDeviceContext, T, 4> trans;
    std::vector<int> axis = {0, 2, 3, 1};
    trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
    trans(dev_ctx, *scores, &scores_swap, axis);

    framework::LoD lod;
    lod.resize(1);
    auto &lod0 = lod[0];
    lod0.push_back(0);
    anchors.Resize({anchors.numel() / 5, 5});
    variances.Resize({variances.numel() / 5, 5});

    int64_t num_proposals = 0;
    for (int64_t i = 0; i < num; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 5, 5});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      std::pair<Tensor, Tensor> tensor_pair =
          ProposalForOneImage(dev_ctx,
                              im_info_slice,
                              anchors,
                              variances,
                              bbox_deltas_slice,
                              scores_slice,
                              pre_nms_top_n,
                              post_nms_top_n,
                              nms_thresh,
                              min_size);
      Tensor &proposals = tensor_pair.first;
      Tensor &scores = tensor_pair.second;

      RRPNAppendProposals(rpn_rois, 5 * num_proposals, proposals);
      RRPNAppendProposals(rpn_roi_probs, num_proposals, scores);
      num_proposals += proposals.dims()[0];
      lod0.push_back(num_proposals);
    }
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 5});
    rpn_roi_probs->Resize({num_proposals, 1});
  }

  std::pair<Tensor, Tensor> ProposalForOneImage(
      const platform::CPUDeviceContext &ctx,
      const Tensor &im_info_slice,
      const Tensor &anchors,
      const Tensor &variances,
      const Tensor &bbox_deltas_slice,  // [M, 5]
      const Tensor &scores_slice,       // [N, 1]
      int pre_nms_top_n,
      int post_nms_top_n,
      float nms_thresh,
      float min_size) const {
    auto *scores_data = scores_slice.data<T>();
    // Sort index
    Tensor index_t;
    index_t.Resize({scores_slice.numel()});
    int *index = index_t.mutable_data<int>(ctx.GetPlace());
    for (int i = 0; i < scores_slice.numel(); ++i) {
      index[i] = i;
    }
    auto compare = [scores_data](const int64_t &i, const int64_t &j) {
      return scores_data[i] > scores_data[j];
    };

    if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
      std::sort(index, index + scores_slice.numel(), compare);
    } else {
      std::nth_element(
          index, index + pre_nms_top_n, index + scores_slice.numel(), compare);
      index_t.Resize({pre_nms_top_n});
    }

    Tensor scores_sel, bbox_sel, anchor_sel, var_sel;
    scores_sel.mutable_data<T>({index_t.numel(), 1}, ctx.GetPlace());
    bbox_sel.mutable_data<T>({index_t.numel(), 5}, ctx.GetPlace());
    anchor_sel.mutable_data<T>({index_t.numel(), 5}, ctx.GetPlace());
    var_sel.mutable_data<T>({index_t.numel(), 5}, ctx.GetPlace());

    CPUGather<T>(ctx, scores_slice, index_t, &scores_sel);
    CPUGather<T>(ctx, bbox_deltas_slice, index_t, &bbox_sel);
    CPUGather<T>(ctx, anchors, index_t, &anchor_sel);
    CPUGather<T>(ctx, variances, index_t, &var_sel);

    auto *scores_ = scores_sel.data<T>();

    Tensor proposals;
    proposals.mutable_data<T>({index_t.numel(), 5}, ctx.GetPlace());
    RBoxCoder<T>(ctx, &anchor_sel, &bbox_sel, &var_sel, &proposals);

    Tensor keep;
    RFilterBoxes<T>(ctx, &proposals, min_size, im_info_slice, &keep);
    Tensor scores_filter;
    bbox_sel.mutable_data<T>({keep.numel(), 5}, ctx.GetPlace());
    scores_filter.mutable_data<T>({keep.numel(), 1}, ctx.GetPlace());
    CPUGather<T>(ctx, proposals, keep, &bbox_sel);
    CPUGather<T>(ctx, scores_sel, keep, &scores_filter);
    if (nms_thresh <= 0) {
      return std::make_pair(bbox_sel, scores_filter);
    }
    Tensor keep_nms = RNMS<T>(ctx, &bbox_sel, &scores_filter, nms_thresh);

    if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
      keep_nms.Resize({post_nms_top_n});
    }
    proposals.mutable_data<T>({keep_nms.numel(), 5}, ctx.GetPlace());
    scores_sel.mutable_data<T>({keep_nms.numel(), 1}, ctx.GetPlace());
    CPUGather<T>(ctx, bbox_sel, keep_nms, &proposals);
    CPUGather<T>(ctx, scores_filter, keep_nms, &scores_sel);

    return std::make_pair(proposals, scores_sel);
  }
};

class RRPNGenerateProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("Scores",
             "(Tensor) The scores from conv is in shape (N, A, H, W), "
             "N is batch size, A is number of anchors, "
             "H and W are height and width of the feature map");
    AddInput("BboxDeltas",
             "(Tensor) Bounding box deltas from conv is in "
             "shape (N, 5*A, H, W).");
    AddInput("ImInfo",
             "(Tensor) Information for image reshape is in shape (N, 3), "
             "in format (height, width, scale)");
    AddInput("Anchors",
             "(Tensor) Bounding box anchors from anchor_generator_op "
             "is in shape (A, H, W, 5).");
    AddInput("Variances",
             "(Tensor) Bounding box variances with same shape as `Anchors`.");

    AddOutput("RpnRois",
              "(LoDTensor), Output proposals with shape (rois_num, 5).");
    AddOutput("RpnRoiProbs",
              "(LoDTensor) Scores of proposals with shape (rois_num, 1).");
    AddAttr<int>("pre_nms_topN",
                 "Number of top scoring RPN proposals to keep before "
                 "applying NMS.");
    AddAttr<int>("post_nms_topN",
                 "Number of top scoring RPN proposals to keep after "
                 "applying NMS");
    AddAttr<float>("nms_thresh", "NMS threshold used on RPN proposals.");
    AddAttr<float>("min_size",
                   "Proposal height and width both need to be greater "
                   "than this min_size.");
    AddComment(R"DOC(
This operator Generate bounding box proposals for Faster RCNN.
The propoasls are generated for a list of images based on image
score 'Scores', bounding box regression result 'BboxDeltas' as
well as predefined bounding box shapes 'anchors'. Greedy
non-maximum suppression is applied to generate the final bounding
boxes.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    rrpn_generate_proposals,
    ops::RRPNGenerateProposalsOp,
    ops::RRPNGenerateProposalsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(rrpn_generate_proposals,
                       ops::RRPNGenerateProposalsKernel<float>,
                       ops::RRPNGenerateProposalsKernel<double>);
