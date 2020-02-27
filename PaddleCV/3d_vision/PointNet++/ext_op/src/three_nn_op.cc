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

class ThreeNNOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ThreeNNOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Known"),
                   "Input(Known) of ThreeNNOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Distance"),
                   "Output(Distance) of ThreeNNOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Idx"),
                   "Output(Idx) of ThreeNNOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");  // [B, N, 3]
    PADDLE_ENFORCE_EQ(dim_x.size(), 3, "X's dimension must be 3");
    PADDLE_ENFORCE_EQ(dim_x[2], 3, "X dim[2] must be 3");

    auto dim_known = ctx->GetInputDim("Known");  // [B, M, 3]
    PADDLE_ENFORCE_EQ(dim_known.size(), 3, "Known's dimension must be 3");
    PADDLE_ENFORCE_EQ(dim_known[2], 3, "Known dim[2] must be 3");

    PADDLE_ENFORCE_EQ(
        dim_x[0], dim_known[0], "X and Known dim[0] should be equal.");
    PADDLE_ENFORCE_GE(
        dim_known[1], 3, "Known dim[1] shoule be greater or euqal than 3.");

    ctx->SetOutputDim("Distance", dim_x);
    ctx->SetOutputDim("Idx", dim_x);
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class ThreeNNOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "The input tensor of three_nn operator. "
             "This is a 3-D tensor with shape of [B, N, 3].");
    AddInput("Known",
             "The input tensor of known points of three_nn operator. "
             "This is a 3-D tensor with shape of [B, M, 3].");
    AddOutput("Distance",
              "The output distance tensor of three_nn operator. "
              "This is a 3-D tensor with shape of [B, N, 3].");
    AddOutput("Idx",
              "The output index tensor of three_nn operator. "
              "This is a 3-D tensor with shape of [B, N, 3].");

    AddAttr<float>("eps", "minimum value of distance.").SetDefault(1e-10);

    AddComment(R"DOC(
          This operator samples the top-3 nearest neighbor of each point
          coordinates specified by Input(X) between known point coordinates
          specified by Input(Known) and calcualte the distance between these
          nearest neighbors.
         )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(three_nn, ops::ThreeNNOp, ops::ThreeNNOpMaker);
