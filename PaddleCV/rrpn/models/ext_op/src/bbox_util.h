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

#pragma once
#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

#define PI 3.141592654

struct RangeInitFunctor {
  int start;
  int delta;
  int* out;
  HOSTDEVICE void operator()(size_t i) { out[i] = start + i * delta; }
};


// get trangle area after  decompose intersecting polygons into triangles
template <typename T>
inline T trangle_area(T* a, T* b, T* c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0;
}

// get area of intersecting
template <typename T>
inline T get_area(T* int_pts, int num_of_inter) {
  T area = 0.0;
  for (int i = 0; i < num_of_inter - 2; i++) {
    area += fabs(
        trangle_area<T>(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}

// sort points to decompose intersecting polygons into triangles
template <typename T>
inline void reorder_pts(T* int_pts, int num_of_inter) {
  if (num_of_inter > 0) {
    T center[2] = {0.0, 0.0};

    for (int i = 0; i < num_of_inter; i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    T vs[16];
    T v[2];
    T d;
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

// determine if points intersect
template <typename T>
inline bool inter2line(T* pts1, T* pts2, int i, int j, T* temp_pts) {
  T a[2] = {pts1[2 * i], pts1[2 * i + 1]};
  T b[2] = {pts1[2 * ((i + 1) % 4)], pts1[2 * ((i + 1) % 4) + 1]};
  T c[2] = {pts2[2 * j], pts2[2 * j + 1]};
  T d[2] = {pts2[2 * ((j + 1) % 4)], pts2[2 * ((j + 1) % 4) + 1]};

  T area_abc, area_abd, area_cda, area_cdb;

  area_abc = trangle_area<T>(a, b, c);
  area_abd = trangle_area<T>(a, b, d);

  if (area_abc * area_abd >= -1e-5) {
    return false;
  }

  area_cda = trangle_area<T>(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= -1e-5) {
    return false;
  }
  T t = area_cda / (area_abd - area_abc);

  T dx = t * (b[0] - a[0]);
  T dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

template <typename T>
inline bool inrect(T pt_x, T pt_y, T* pts) {
  T ab[2] = {pts[2] - pts[0], pts[3] - pts[1]};
  T ad[2] = {pts[6] - pts[0], pts[7] - pts[1]};
  T ap[2] = {pt_x - pts[0], pt_y - pts[1]};

  T abab = ab[0] * ab[0] + ab[1] * ab[1];
  T abap = ab[0] * ap[0] + ab[1] * ap[1];
  T adad = ad[0] * ad[0] + ad[1] * ad[1];
  T adap = ad[0] * ap[0] + ad[1] * ap[1];
  bool result = (abab - abap >= -1) and (abap >= -1) and (adad - adap >= -1) and
                (adap >= -1);
  return result;
}

// calculate the number of intersection points
template <typename T>
inline int inter_pts(T* pts1, T* pts2, T* int_pts) {
  int num_of_inter = 0;

  for (int i = 0; i < 4; i++) {
    if (inrect<T>(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
    if (inrect<T>(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  T out_pts[2];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      bool has_pts = inter2line<T>(pts1, pts2, i, j, out_pts);
      if (has_pts) {
        int_pts[num_of_inter * 2] = out_pts[0];
        int_pts[num_of_inter * 2 + 1] = out_pts[1];
        num_of_inter++;
      }
    }
  }


  return num_of_inter;
}

// convert x,y,w,h,angle to x1,y1,x2,y2,x3,y3,x4,y4
template <typename T>
inline void convert_region(T* pts,
                           const framework::Tensor& _region,
                           int index) {
  auto region = framework::EigenTensor<T, 2>::From(_region);
  T angle = region(index, 4);
  T a_cos = cos(angle / 180.0 * PI);
  T a_sin = -sin(angle / 180.0 * PI);  // anti clock-wise

  T ctr_x = region(index, 0);
  T ctr_y = region(index, 1);
  T h = region(index, 3);
  T w = region(index, 2);


  T pts_x[4] = {-w / 2, -w / 2, w / 2, w / 2};
  T pts_y[4] = {-h / 2, h / 2, h / 2, -h / 2};

  for (int i = 0; i < 4; i++) {
    pts[2 * i] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    pts[2 * i + 1] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;
  }
}


// Calculate the area of intersection
template <typename T>
inline float inter(const framework::Tensor& _region1,
                   const framework::Tensor& _region2,
                   const int& r,
                   const int& c) {
  T pts1[8];
  T pts2[8];
  T int_pts[16];
  int num_of_inter;


  convert_region<T>(pts1, _region1, r);
  convert_region<T>(pts2, _region2, c);

  num_of_inter = inter_pts<T>(pts1, pts2, int_pts);

  reorder_pts<T>(int_pts, num_of_inter);

  return get_area<T>(int_pts, num_of_inter);
}

template <typename T>
inline float devRotateIoU(const framework::Tensor& _region1,
                          const framework::Tensor& _region2,
                          const int r,
                          const int c) {
  auto __region1 = framework::EigenTensor<T, 2>::From(_region1);
  auto __region2 = framework::EigenTensor<T, 2>::From(_region2);

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
  auto result = area_inter / (area1 + area2 - area_inter);

  if (result < 0) {
    result = 0.0;
  }
  // may have bugs which cause overlap > 1
  if (result > 1.00000001) {
    result = 0.0;
  }
  return result;
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
    trg(i, 4) = (PI / 180) * trg(i, 4);
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


}  // namespace operators
}  // namespace paddle
