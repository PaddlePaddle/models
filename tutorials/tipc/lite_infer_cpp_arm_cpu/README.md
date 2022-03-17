# Lite ARM CPU 推理开发文档

# 目录

- [1. 简介](#1)
- [2. Lite ARM CPU 推理功能开发与规范](#2)
    - [2.1 开发流程](#2.1)
    - [2.2 核验点](#2.2)
- [3. Lite ARM CPU 推理测试开发与规范](#3)
    - [3.1 开发流程](#3.1)
    - [3.2 核验点](#3.2)


<a name="1"></a>

## 1. 简介

该系列文档主要介绍飞桨模型基于 Linux ARM CPU 推理开发过程，主要包含2个步骤。


- 步骤一：参考[《Lite ARM CPU 推理开发文档》](./lite_infer_cpp_arm_cpu.md)，完成基于Paddle-Lite的Lite ARM CPU 推理程序开发。

- 步骤二：参考[《Lite ARM CPU 推理测试开发文档》](./test_lite_infer_cpp_arm_cpu.md)，完成Lite ARM CPU 推理测试程序开发。


<a name="2"></a>

## 2. Lite ARM CPU 推理功能开发与规范

### 2.1 开发流程

Lite ARM CPU 推理开发过程可以分为下面7个步骤。

<div align="center">
    <img src="../images/lite_infer_arm_cpu_guide.png" width="800">
</div>


更多的介绍可以参考：[Lite ARM CPU 推理开发文档](./lite_infer_cpp_arm_cpu.md)。

<a name="2.2"></a>

### 2.2 核验点

#### 2.2.1 准备推理模型

* 使用Paddle-Lite提供的opt工具进行转换之后，需要生成后缀为`*.nb`的推理模型。

#### 2.2.2 验证推理结果正确性

* 对于相同的输入，Lite ARM CPU 推理结果需要与Paddle Inference推理结果相同。**如果预处理或者后处理部分无法保持一致，导致输出结果有不同，需要给出详细的原因说明**。

<a name="3"></a>

## 3. Lite ARM CPU 推理测试开发与规范

<a name="3.1"></a>

### 3.1 开发流程

基础训练推理测试开发的流程如下所示。

<div align="center">
    <img src="./images/test_lite_infer_cpp_arm_cpu_pipeline.png" width="400">
</div>


更多的介绍可以参考：[Lite cpp arm cpu 推理功能测试开发文档](./test_lite_infer_cpp_arm_cpu.md)。


<a name="3.2"></a>

### 3.2 核验点

#### 3.2.1 目录结构

在repo根目录下面新建`test_tipc`文件夹，目录结构如下所示。

```
test_tipc
    |--configs                              # 配置目录
    |    |--model_name                      # 您的模型名称
    |           |--lite_arm_cpu_cpp.txt     # lite_infer_cpp_arm_cpu推理测试配置文件
    |--docs                                 # 文档目录
    |   |--test_lite_infer_cpp_arm_cpu.md   # lite_infer_cpp_arm_cpu推理测试说明文档
    |----README.md                          # TIPC说明文档
    |----prepare_lite_arm_cpu_cpp.sh        # TIPC lite_infer_cpp_arm_cpu推理数据、模型、预测库准备脚本
    |----test_lite_arm_cpu_cpp.sh           # TIPC lite_infer_cpp_arm_cpu推理测试解析脚本，无需改动
```


#### 3.2.2 配置文件和测试文档

1. TIPC功能总览文档：`test_tipc/README.md`
2. Lite cpp arm cpu 推理功能测试说明文档：`test_tipc/docs/test_lite_infer_cpp_arm_cpu.md`
