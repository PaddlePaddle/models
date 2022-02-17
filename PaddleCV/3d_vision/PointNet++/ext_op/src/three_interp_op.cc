/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

using framework::Tensor;

class ThreeInterpOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ThreeInterpOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(Weight) of ThreeInterpOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Idx"),
                   "Input(Idx) of ThreeInterpOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ThreeInterpOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");  // [B, M, C]
    PADDLE_ENFORCE_EQ(dim_x.size(), 3, "X's dimension must be 3");

    auto dim_weight = ctx->GetInputDim("Weight");  // [B, N, 3]
    PADDLE_ENFORCE_EQ(dim_weight.size(), 3, "Weight's dimension must be 3");

    PADDLE_ENFORCE_EQ(
        dim_x[0], dim_weight[0], "X and Weight dim[0] should be equal.");

    auto dim_idx = ctx->GetInputDim("Idx");  // [B, N, 3]
    PADDLE_ENFORCE_EQ(dim_idx.size(), 3, "Idx's dimension must be 3");

    for (int i = 0; i < 3; i++) {
      PADDLE_ENFORCE_EQ(
          dim_weight[i], dim_idx[i], "Weight and Idx shape should be same.");
    }

    // output: [B, N, C]
    std::vector<int64_t> dim_out({dim_x[0], dim_idx[1], dim_x[2]});
    ctx->SetOutputDim("Out", framework::make_ddim(dim_out));
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class ThreeInterpOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "The input tensor of three_interp operator. "
             "This is a 3-D tensor with shape of [B, M, C].");
    AddInput("Weight",
             "The input tensor of point weight of three_interp operator. "
             "This is a 3-D tensor with shape of [B, N, 3].");
    AddInput("Idx",
             "The input tensor of nearest neighbor index of three_interp "
             "operator. This is a 3-D tensor with shape of [B, N, 3].");
    AddOutput("Out",
              "The output tensor of three_interp operator. "
              "This is a 3-D tensor with shape of [B, N, 3].");

    AddComment(R"DOC(
          This operator calculate interpolate results from input, weight and
          index.
         )DOC");
  }
};

class ThreeInterpOpGrad : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Weight"), "Input(Weight) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Idx"), "Input(Idx) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

template <typename T>
class ThreeInterpGradDescMaker : public framework::SingleGradOpMaker<T> {
public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("three_interp_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Weight", this->Input("Weight"));
    op->SetInput("Idx", this->Input("Idx"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(three_interp,
                  ops::ThreeInterpOp,
                  ops::ThreeInterpOpMaker,
                  ops::ThreeInterpGradDescMaker<paddle::framework::OpDesc>,
                  ops::ThreeInterpGradDescMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(three_interp_grad, ops::ThreeInterpOpGrad);
