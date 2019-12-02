/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

struct RangeInitFunctor {
  int start;
  int delta;
  int* out;
  HOSTDEVICE void operator()(size_t i) { out[i] = start + i * delta; }
};


template <typename T>
inline T trangle_area(T* a, T* b, T* c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0;
}


template <typename T>
inline T area(T* int_pts, int num_of_inter) {
  float area = 0.0;
  for (int i = 0; i < num_of_inter - 2; i++) {
    area +=
        fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}

template <typename T>
inline void reorder_pts(T* int_pts, int num_of_inter) {
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
inline bool inter2line(T* pts1, T* pts2, int i, int j, T* temp_pts) {
  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

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

  if (area_abc * area_abd >= -1e-5) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= -1e-5) {
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
inline bool inrect(T pt_x, T pt_y, T* pts) {
  double ab[2];
  double ad[2];
  double ap[2];

  double abab;
  double abap;
  double adad;
  double adap;

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
  bool result = (abab - abap >= -1) and (abap >= -1) and (adad - adap >= -1) and
                (adap >= -1);
  return result;
}


template <typename T>
inline int inter_pts(T* pts1, T* pts2, T* int_pts) {
  int num_of_inter = 0;

  for (int i = 0; i < 4; i++) {
    if (inrect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
    if (inrect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
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
inline void convert_region(T* pts,
                           const framework::Tensor& _region,
                           int index) {
  auto region = framework::EigenTensor<T, 2>::From(_region);
  float angle = region(index, 4);
  float a_cos = cos(angle / 180.0 * 3.1415926535);
  float a_sin = -sin(angle / 180.0 * 3.1415926535);  // anti clock-wise

  float ctr_x = region(index, 0);
  float ctr_y = region(index, 1);
  float h = region(index, 3);
  float w = region(index, 2);


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
inline float inter(const framework::Tensor& _region1,
                   const framework::Tensor& _region2,
                   const int& r,
                   const int& c) {
  T pts1[8];
  T pts2[8];
  T int_pts[16];
  int num_of_inter;


  convert_region(pts1, _region1, r);
  convert_region(pts2, _region2, c);

  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);
}

template <typename T>
inline float devRotateIoU(const framework::Tensor& _region1,
                          const framework::Tensor& _region2,
                          const int r,
                          const int c) {
  auto __region1 = framework::EigenTensor<T, 2>::From(_region1);
  auto __region2 = framework::EigenTensor<T, 2>::From(_region2);
  // auto __region1 = _region1;
  // auto __region2 = _region2;

  if ((fabs(__region1(r, 0) - __region2(c, 0)) < 1e-5) &&
      (fabs(__region1(r, 1) - __region2(c, 1)) < 1e-5) &&
      (fabs(__region1(r, 2) - __region2(c, 2)) < 1e-5) &&
      (fabs(__region1(r, 3) - __region2(c, 3)) < 1e-5) &&
      (fabs(__region1(r, 4) - __region2(c, 4)) < 1e-5)) {
    return 1.0;
  }
  T area1, area2, area_inter;
  area1 = __region1(r, 2) * __region1(r, 3);
  area2 = __region2(c, 2) * __region2(c, 3);
  area_inter = inter<T>(_region1, _region2, r, c);
  // std::cout <<  "area1:" <<  area1 << std::endl;
  // std::cout <<  "area2:" <<  area2 << std::endl;
  // std::cout <<  "area_inter:" <<  area_inter << std::endl;
  auto result = area_inter / (area1 + area2 - area_inter);

  if (result < 0) {
    result = 0.0;
  }
  if (result > 1.00000001) {
    result = 0.0;
  }
  return result;
}


template <typename T>
inline HOSTDEVICE T RoIArea(const T* box, bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}


template <typename T>
inline void BoxToDelta2(const int box_num,
                        const framework::Tensor& ex_boxes,
                        const framework::Tensor& gt_boxes,
                        const float* weights,
                        framework::Tensor* box_delta) {
  auto ex_boxes_et = framework::EigenTensor<T, 2>::From(ex_boxes);
  auto gt_boxes_et = framework::EigenTensor<T, 2>::From(gt_boxes);
  auto trg = framework::EigenTensor<T, 2>::From(*box_delta);
  T ex_w, ex_h, ex_ctr_x, ex_ctr_y, ex_angle, gt_w, gt_h, gt_ctr_x, gt_ctr_y,
      gt_angle;
  for (int64_t i = 0; i < box_num; ++i) {
    ex_w = ex_boxes_et(i, 2);
    ex_h = ex_boxes_et(i, 3);
    ex_ctr_x = ex_boxes_et(i, 0);
    ex_ctr_y = ex_boxes_et(i, 1);
    ex_angle = ex_boxes_et(i, 4);

    gt_w = gt_boxes_et(i, 2);
    gt_h = gt_boxes_et(i, 3);
    gt_ctr_x = gt_boxes_et(i, 0);
    gt_ctr_y = gt_boxes_et(i, 1);
    gt_angle = gt_boxes_et(i, 4);

    trg(i, 0) = (gt_ctr_x - ex_ctr_x) / ex_w;
    trg(i, 1) = (gt_ctr_y - ex_ctr_y) / ex_h;
    trg(i, 2) = std::log(gt_w / ex_w);
    trg(i, 3) = std::log(gt_h / ex_h);
    trg(i, 4) = gt_angle - ex_angle;

    if (weights) {
      trg(i, 0) = trg(i, 0) * weights[0];
      trg(i, 1) = trg(i, 1) * weights[1];
      trg(i, 2) = trg(i, 2) * weights[2];
      trg(i, 3) = trg(i, 3) * weights[3];
      trg(i, 4) = trg(i, 4) * weights[4];
    }

    if (gt_angle <= -30 && ex_angle >= 120) {
      trg(i, 4) = trg(i, 4) + 180.0;
    }
    if (gt_angle >= 120 && ex_angle <= -30) {
      trg(i, 4) = trg(i, 4) - 180.0;
    }
    trg(i, 4) = (3.14159265358979323846264338327950288 / 180) * trg(i, 4);
  }
}


/*
 * transform that computes target bounding-box regression deltas
 * given proposal boxes and ground-truth boxes.
 */
template <typename T>
inline void BoxToDelta(const int box_num,
                       const framework::Tensor& ex_boxes,
                       const framework::Tensor& gt_boxes,
                       const float* weights,
                       const bool normalized,
                       framework::Tensor* box_delta) {
  auto ex_boxes_et = framework::EigenTensor<T, 2>::From(ex_boxes);
  auto gt_boxes_et = framework::EigenTensor<T, 2>::From(gt_boxes);
  auto trg = framework::EigenTensor<T, 2>::From(*box_delta);
  T ex_w, ex_h, ex_ctr_x, ex_ctr_y, gt_w, gt_h, gt_ctr_x, gt_ctr_y;
  for (int64_t i = 0; i < box_num; ++i) {
    ex_w = ex_boxes_et(i, 2) - ex_boxes_et(i, 0) + (normalized == false);
    ex_h = ex_boxes_et(i, 3) - ex_boxes_et(i, 1) + (normalized == false);
    ex_ctr_x = ex_boxes_et(i, 0) + 0.5 * ex_w;
    ex_ctr_y = ex_boxes_et(i, 1) + 0.5 * ex_h;

    gt_w = gt_boxes_et(i, 2) - gt_boxes_et(i, 0) + (normalized == false);
    gt_h = gt_boxes_et(i, 3) - gt_boxes_et(i, 1) + (normalized == false);
    gt_ctr_x = gt_boxes_et(i, 0) + 0.5 * gt_w;
    gt_ctr_y = gt_boxes_et(i, 1) + 0.5 * gt_h;

    trg(i, 0) = (gt_ctr_x - ex_ctr_x) / ex_w;
    trg(i, 1) = (gt_ctr_y - ex_ctr_y) / ex_h;
    trg(i, 2) = std::log(gt_w / ex_w);
    trg(i, 3) = std::log(gt_h / ex_h);

    if (weights) {
      trg(i, 0) = trg(i, 0) / weights[0];
      trg(i, 1) = trg(i, 1) / weights[1];
      trg(i, 2) = trg(i, 2) / weights[2];
      trg(i, 3) = trg(i, 3) / weights[3];
    }
  }
}

template <typename T>
void Gather(
    const T* in, const int in_stride, const int* index, const int num, T* out) {
  const int stride_bytes = in_stride * sizeof(T);
  for (int i = 0; i < num; ++i) {
    int id = index[i];
    memcpy(out + i * in_stride, in + id * in_stride, stride_bytes);
  }
}

template <typename T>
void BboxOverlaps2(const framework::Tensor& r_boxes,
                   const framework::Tensor& c_boxes,
                   framework::Tensor* overlaps) {
  auto overlaps_et = framework::EigenTensor<T, 2>::From(*overlaps);
  int r_num = r_boxes.dims()[0];
  int c_num = c_boxes.dims()[0];
  for (int i = 0; i < r_num; ++i) {
    for (int j = 0; j < c_num; ++j) {
      overlaps_et(i, j) = devRotateIoU<T>(r_boxes, c_boxes, i, j);
    }
  }
}

template <typename T>
void BboxOverlaps(const framework::Tensor& r_boxes,
                  const framework::Tensor& c_boxes,
                  framework::Tensor* overlaps) {
  auto r_boxes_et = framework::EigenTensor<T, 2>::From(r_boxes);
  auto c_boxes_et = framework::EigenTensor<T, 2>::From(c_boxes);
  auto overlaps_et = framework::EigenTensor<T, 2>::From(*overlaps);
  int r_num = r_boxes.dims()[0];
  int c_num = c_boxes.dims()[0];
  auto zero = static_cast<T>(0.0);
  T r_box_area, c_box_area, x_min, y_min, x_max, y_max, inter_w, inter_h,
      inter_area;
  for (int i = 0; i < r_num; ++i) {
    r_box_area = (r_boxes_et(i, 2) - r_boxes_et(i, 0) + 1) *
                 (r_boxes_et(i, 3) - r_boxes_et(i, 1) + 1);
    for (int j = 0; j < c_num; ++j) {
      c_box_area = (c_boxes_et(j, 2) - c_boxes_et(j, 0) + 1) *
                   (c_boxes_et(j, 3) - c_boxes_et(j, 1) + 1);
      x_min = std::max(r_boxes_et(i, 0), c_boxes_et(j, 0));
      y_min = std::max(r_boxes_et(i, 1), c_boxes_et(j, 1));
      x_max = std::min(r_boxes_et(i, 2), c_boxes_et(j, 2));
      y_max = std::min(r_boxes_et(i, 3), c_boxes_et(j, 3));
      inter_w = std::max(x_max - x_min + 1, zero);
      inter_h = std::max(y_max - y_min + 1, zero);
      inter_area = inter_w * inter_h;
      overlaps_et(i, j) =
          (inter_area == 0.) ? 0 : inter_area /
                                       (r_box_area + c_box_area - inter_area);
    }
  }
}


template <class T>
void ClipTiledBoxes(const platform::DeviceContext& ctx,
                    const framework::Tensor& im_info,
                    const framework::Tensor& input_boxes,
                    framework::Tensor* out) {
  T* out_data = out->mutable_data<T>(ctx.GetPlace());
  const T* im_info_data = im_info.data<T>();
  const T* input_boxes_data = input_boxes.data<T>();
  T zero(0);
  T im_w = round(im_info_data[1] / im_info_data[2]);
  T im_h = round(im_info_data[0] / im_info_data[2]);
  for (int64_t i = 0; i < input_boxes.numel(); ++i) {
    if (i % 4 == 0) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_w - 1), zero);
    } else if (i % 4 == 1) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_h - 1), zero);
    } else if (i % 4 == 2) {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_w - 1), zero);
    } else {
      out_data[i] = std::max(std::min(input_boxes_data[i], im_h - 1), zero);
    }
  }
}

}  // namespace operators
}  // namespace paddle
