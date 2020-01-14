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

#include <algorithm>
#include <limits>
#include <memory>
#include "math_function.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class RRPNRotatedROIAlignOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Rotated ROIAlignOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("ROIs"),
                   "Input(ROIs) of Rotated ROIAlignOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of Rotated ROIAlignOp should not be null.");
    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    PADDLE_ENFORCE(input_dims.size() == 4,
                   "The format of input tensor is NCHW.");
    PADDLE_ENFORCE(rois_dims.size() == 2,
                   "ROIs should be a 2-D LoDTensor of shape (num_rois, 5)"
                   "given as [[x1, y1, x2, y2, theta], ...].");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE(rois_dims[1] == 5,
                     "ROIs should be a 2-D LoDTensor of shape (num_rois, 5)"
                     "given as [[x1, y1, x2, y2, theta], ...].");
    }
    int pooled_height = ctx->Attrs().Get<int>("pooled_height");
    int pooled_width = ctx->Attrs().Get<int>("pooled_width");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");

    PADDLE_ENFORCE_GT(
        pooled_height, 0, "The pooled output height must greater than 0");
    PADDLE_ENFORCE_GT(
        pooled_width, 0, "The pooled output width must greater than 0");
    PADDLE_ENFORCE_GT(
        spatial_scale, 0.0f, "The spatial scale must greater than 0");

    auto out_dims = input_dims;
    out_dims[0] = rois_dims[0];
    out_dims[1] = input_dims[1];
    out_dims[2] = pooled_height;
    out_dims[3] = pooled_width;

    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("ConIdX", out_dims);
    ctx->SetOutputDim("ConIdY", out_dims);
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class RRPNRotatedROIAlignGradOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "The GRAD@Out of RotatedROIAlignGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName("X")),
                   "The GRAD@X of RotatedROIAlignGradOp should not be null.");
    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("ROIs")->type(),
                                   ctx.device_context());
  }
};

class RRPNRotatedROIAlignOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "The input of RRPNRotatedROIAlignOp. The data type is float32 or "
             "float64."
             "The format of input tensor is NCHW. Where N is batch size, "
             "C is the number of input channels, "
             "H is the height of the feature, and "
             "W is the width of the feature.");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to pool over. "
             "should be a 2-D LoDTensor of shape (num_rois, 5)"
             "given as [[x, y, w, h, theta], ...]. "
             "(x, y) is the center coordinates, and "
             "(w, h) is the bottom right coordinates, theta is rotation angle"
             "of ROI.");
    AddOutput("Out",
              "(Tensor), "
              "The output of ROIAlignOp is a 4-D tensor with shape "
              "(num_rois, channels, pooled_h, pooled_w). The data type is "
              "float32 or float64.");
    AddOutput("ConIdX",
              "(Tensor), "
              "index x of affine transform");
    AddOutput("ConIdY",
              "(Tensor), "
              "index y of affine transform");

    AddAttr<float>("spatial_scale",
                   "(float, default 1.0), "
                   "Multiplicative spatial scale factor "
                   "to translate ROI coords from their input scale "
                   "to the scale used when pooling.")
        .SetDefault(1.0);
    AddAttr<int>("pooled_height",
                 "(int, default 1), "
                 "The pooled output height.")
        .SetDefault(1);
    AddAttr<int>("pooled_width",
                 "(int, default 1), "
                 "The pooled output width.")
        .SetDefault(1);
    AddComment(R"DOC(
**RotatedRoIAlign Operator**

Rotated Region of interest align (also known as Rotated RoI align) is to perform
bilinear interpolation on inputs of nonuniform sizes to obtain 
fixed-size feature maps (e.g. 7*7)

Dividing each region proposal into equal-sized sections with
the pooled_width and pooled_height. Location remains the origin
result.

In each ROI bin, the value of the four regularly sampled locations 
are computed directly through bilinear interpolation. The output is
the mean of four locations.
Thus avoid the misaligned problem.   
    )DOC");
  }
};

template <typename T>
class RRPNRotatedROIAlignGradMaker : public framework::SingleGradOpMaker<T> {
public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T);
    op->SetType("rrpn_rotated_roi_align_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("ConIdX", this->Output("ConIdX"));
    op->SetInput("ConIdY", this->Output("ConIdY"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(
    RRPNRotatedRoiAlignGradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    rrpn_rotated_roi_align,
    ops::RRPNRotatedROIAlignOp,
    ops::RRPNRotatedROIAlignOpMaker,
    ops::RRPNRotatedROIAlignGradMaker<paddle::framework::OpDesc>,
    ops::RRPNRotatedROIAlignGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(rrpn_rotated_roi_align_grad,
                  ops::RRPNRotatedROIAlignGradOp,
                  ops::RRPNRotatedRoiAlignGradNoNeedBufVarsInferer);
