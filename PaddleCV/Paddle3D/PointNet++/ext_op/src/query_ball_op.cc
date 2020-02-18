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

#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class QueryBallOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // points: [b,n,3]
    PADDLE_ENFORCE(ctx->HasInput("Points"), "Input(Points) shoud not be null");
    auto p_dims = ctx->GetInputDim("Points");
    PADDLE_ENFORCE(p_dims.size() == 3 && p_dims[2] == 3,
                   "Input(Points) of QueryBallOp should be 3-D Tensor, the "
                   "last dimension must be 3");
    // new_points: [b,m,3]
    PADDLE_ENFORCE(ctx->HasInput("New_Points"),
                   "Input(New_Points) shoud not be null");
    auto np_dims = ctx->GetInputDim("New_Points");
    PADDLE_ENFORCE(np_dims.size() == 3 && np_dims[2] == 3,
                   "Input(New_Points) of QueryBallOp should be 3-D Tensor, the "
                   "last dimension must be 3");
    int n_sample = ctx->Attrs().Get<int>("N_sample");
    PADDLE_ENFORCE(n_sample >= 0,
                   "The n_sample should be greater than or equal to 0.");
    float radius = ctx->Attrs().Get<float>("Radius");
    PADDLE_ENFORCE(radius >= 0,
                   "The radius should be greater than or equal to 0.");
    // output: [b,m,nsample]
    std::vector<int64_t> dim_out({p_dims[0], np_dims[1], n_sample});
    ctx->SetOutputDim("Output", framework::make_ddim(dim_out));
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("Points")->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class QueryBallOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("Points",
             "Input points with shape (batch, n, 3), n is input "
             "points's num");
    AddInput("New_Points",
             "Query points with shape (batch, m, 3), m is query points's num");
    AddOutput("Output", "output points with shape(batch, m, nsample)");
    AddAttr<int>("N_sample",
                 R"Doc(Number of points selected in each ball region")Doc")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<float>("Radius",
                   R"Doc(Ball search radius with shape(1))Doc")
        .SetDefault(0)
        .EqualGreaterThan(0);

    AddComment(
        R"Doc(Query Ball Points)Doc");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(query_ball, ops::QueryBallOp, ops::QueryBallOpMaker);
