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

//#include "rrpn_box_coder_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class RRPNBoxCoderOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("PriorBox"),
                   "Input(PriorBox) of BoxCoderOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("TargetBox"),
                   "Input(TargetBox) of BoxCoderOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("OutputBox"),
                   "Output(OutputBox) of BoxCoderOp should not be null.");

    auto prior_box_dims = ctx->GetInputDim("PriorBox");
    // auto target_box_dims = ctx->GetInputDim("TargetBox");

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          prior_box_dims.size(), 2, "The rank of Input PriorBox must be 2");
      PADDLE_ENFORCE_EQ(
          prior_box_dims[1], 5, "The shape of PriorBox is [N, 5]");
      if (ctx->HasInput("PriorBoxVar")) {
        auto prior_box_var_dims = ctx->GetInputDim("PriorBoxVar");
        PADDLE_ENFORCE(prior_box_var_dims.size() == 2,
                       "Input(PriorBoxVar) of BoxCoderOp should be 2.");
        PADDLE_ENFORCE_EQ(
            prior_box_dims,
            prior_box_var_dims,
            "The dimension of Input(PriorBoxVar) should be equal to"
            "the dimension of Input(PriorBox) when the rank is 2.");
      }
    }
  }
};

class RRPNBoxCoderOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput(
        "PriorBox",
        "(Tensor, default Tensor<float>) "
        "Box list PriorBox is a 2-D Tensor with shape [M, 5] holds M boxes, "
        "each box is represented as [x, y, w, h, angle], "
        "[x, y] is the center coordinate of the anchor box, "
        "if the input is image feature map, they are close to the origin "
        "of the coordinate system. [w, h] is the width and height "
        "of the anchor box, angle is angle of rotation.");
    AddInput("PriorBoxVar",
             "(Tensor, default Tensor<float>, optional) "
             "PriorBoxVar is a 2-D Tensor with shape [M, 5] holds M group "
             "of variance. PriorBoxVar will set all elements to 1 by "
             "default.")
        .AsDispensable();
    AddInput(
        "TargetBox",
        "(LoDTensor or Tensor) This input can be a 2-D LoDTensor with shape "
        "[N, 5], each box is represented as [x, y, w, h, angle],"
        "[x, y] is the center coordinate of the box, [w, h] is width and "
        "height of the box,"
        "angle is angle of rotation around the center of box.");
    AddAttr<std::vector<float>>(
        "variance",
        "(vector<float>, default {}),"
        "variance of prior box with shape [5]. PriorBoxVar and variance can"
        "not be provided at the same time.")
        .SetDefault(std::vector<float>{});
    AddOutput("OutputBox",
              "(Tensor) "
              "2-D Tensor with shape [M, 5] which M represents the number of "
              "deocded boxes"
              "and 5 represents [x, y, w, h, angle]");

    AddComment(R"DOC(

Rotatedi Bounding Box Coder.

Decode the target bounding box with the priorbox information.

The Decoding schema described below:

    ox = pw * tx / pxv + cx

    oy = ph * ty / pyv + cy

    ow = exp(tw / pwv) * pw

    oh = exp(th / phv) * ph

    oa = ta / pav  * 1.0 / 3.141592653 * 180 + pa

where `tx`, `ty`, `tw`, `th`, `ta` denote the target box's center coordinates, width
,height and angle respectively. Similarly, `px`, `py`, `pw`, `ph`, `pa` denote the
priorbox's (anchor) center coordinates, width, height and angle. `pxv`, `pyv`, `pwv`,
`phv`, `pav` denote the variance of the priorbox and `ox`, `oy`, `ow`, `oh`, `oa`
denote the encoded/decoded coordinates, width and height. 
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    rrpn_box_coder,
    ops::RRPNBoxCoderOp,
    ops::RRPNBoxCoderOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
