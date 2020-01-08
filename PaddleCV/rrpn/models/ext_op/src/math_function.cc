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

#include "math_function.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <vector>
#include "math_function_impl.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

#define DEFINE_CPU_TRANS(RANK)                                          \
  template struct Transpose<platform::CPUDeviceContext,                 \
                            platform::float16,                          \
                            RANK>;                                      \
  template struct Transpose<platform::CPUDeviceContext, float, RANK>;   \
  template struct Transpose<platform::CPUDeviceContext, double, RANK>;  \
  template struct Transpose<platform::CPUDeviceContext, int, RANK>;     \
  template struct Transpose<platform::CPUDeviceContext, int64_t, RANK>; \
  template struct Transpose<platform::CPUDeviceContext, bool, RANK>;    \
  template struct Transpose<platform::CPUDeviceContext, int16_t, RANK>; \
  template struct Transpose<platform::CPUDeviceContext, uint8_t, RANK>; \
  template struct Transpose<platform::CPUDeviceContext, int8_t, RANK>;

DEFINE_CPU_TRANS(1);
DEFINE_CPU_TRANS(2);
DEFINE_CPU_TRANS(3);
DEFINE_CPU_TRANS(4);
DEFINE_CPU_TRANS(5);
DEFINE_CPU_TRANS(6);

template <typename DeviceContext, typename T, int Rank>
void Transpose<DeviceContext, T, Rank>::operator()(
    const DeviceContext& context,
    const framework::Tensor& in,
    framework::Tensor* out,
    const std::vector<int>& axis) {
  Eigen::array<int, Rank> permute;
  for (int i = 0; i < Rank; i++) {
    permute[i] = axis[i];
  }
  auto eigen_in = framework::EigenTensor<T, Rank>::From(in);
  auto eigen_out = framework::EigenTensor<T, Rank>::From(*out);
  auto* dev = context.eigen_device();
  eigen_out.device(*dev) = eigen_in.shuffle(permute);
}


}  // namespace math
}  // namespace operators
}  // namespace paddle
