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

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FarthestPointSamplingOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "(Tensor)input point cloud dataset with shape (B, N, 3)"
             "B is batch size, N is points's nums, 3 is (x,y,z) coordinate");
    AddOutput("Output",
              "(Tensor)return sampled points with shape (B, M)"
              "B is batch size, M is points's nums");
    AddAttr<int>("sampled_point_num", "sampling points's num")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(
        R"Doc(
            Sampling point based on 
            its max eucliden distance with other points.)Doc");
  }
};

class FarthestPointSamplingOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shoud not be null");
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(x_dims.size() == 3,
                   "Input(X) of FathestPointSamplingOp should be 3-D Tensor");
    const int m = ctx->Attrs().Get<int>("sampled_point_num");
    ctx->SetOutputDim("Output", {x_dims[0], m});
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("X")->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(farthest_point_sampling,
                  ops::FarthestPointSamplingOp,
                  ops::FarthestPointSamplingOpMaker);
