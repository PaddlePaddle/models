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
#include <vector>
#include "blas.h"
#include "math_function.h"
#include "math_function_impl.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

using float16 = paddle::platform::float16;

template struct SetConstant<platform::CUDADeviceContext, platform::float16>;
template struct SetConstant<platform::CUDADeviceContext, float>;
template struct SetConstant<platform::CUDADeviceContext, double>;
template struct SetConstant<platform::CUDADeviceContext, int>;
template struct SetConstant<platform::CUDADeviceContext, int64_t>;
template struct SetConstant<platform::CUDADeviceContext, bool>;

//#define DEFINE_GPU_TRANS(RANK)                                           \
//  template struct Transpose<platform::CUDADeviceContext, float, RANK>;   \
//  template struct Transpose<platform::CUDADeviceContext, double, RANK>;  \
//  template struct Transpose<platform::CUDADeviceContext, float16, RANK>; \
//  template struct Transpose<platform::CUDADeviceContext, int8_t, RANK>;  \
//  template struct Transpose<platform::CUDADeviceContext, int32_t, RANK>; \
//  template struct Transpose<platform::CUDADeviceContext, int64_t, RANK>;
//
// DEFINE_GPU_TRANS(1);
// DEFINE_GPU_TRANS(2);
// DEFINE_GPU_TRANS(3);
// DEFINE_GPU_TRANS(4);
// DEFINE_GPU_TRANS(5);
// DEFINE_GPU_TRANS(6);

struct TensorSetConstantGPU {
  TensorSetConstantGPU(const platform::DeviceContext& context,
                       framework::Tensor* tensor,
                       float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename T>
  void apply() const {
    SetConstant<platform::CUDADeviceContext, T> functor;
    functor(reinterpret_cast<const platform::CUDADeviceContext&>(context_),
            tensor_,
            static_cast<T>(value_));
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};


}  // namespace math
}  // namespace operators
}  // namespace paddle
