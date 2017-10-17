# CAPI 实现模型预测

## 1. 编译 PaddlePaddle Library
[TBD]

## 2. CAPI 使用流程指南

### 1. 序列化模型配置

使用 CAPI 进行预测时，需要先将网络配置转换为 protobuf 二进制文件。假设网络配置文件为 `trainer_config.conf`，通过调用 PaddlePaddle 中的 `dump_config` 模块可以将配置文件  `trainer_config.conf` 转化为 `trainer_config.bin` 。 执行以下命令完成转换：
```shell
python -m paddle.utils.dump_config trainer_config.py '' --binary > trainer_config.bin
```

### 2. 组织输入数据
在 PaddlePaddle 中， 神经网络的每个输入会被组织成一个 `arguments` 对象。 本单元的示例将详细介绍如何组织不同类型的数据进行预测。

### 3. 初始化/加载模型

#### PaddlePaddle 初始化
在加载模型之前，需要对 PaddlePaddle 进行初始化操作，例如是否使用 GPU 等，示例代码如下 ：
```c
char* argv[] = {"--use_gpu=False"};
paddle_init(1, (char**)argv);
```
#### 加载模型
在 PaddlePaddle 中， `gradient machine` 表示一个可以进行前向计算和后向传播的神经网络结构,我们可以构建一个 `gradient machine` 对象，从磁盘中加载参数进行模型预测。示例代码如下：
```c
paddle_gradient_machine machine;
paddle_gradient_machine_create_for_inference(&machine, config_file_content, content_size));
paddle_gradient_machine_load_parameter_from_disk(machine, "./some_where_to_params"));
```

### 4. 前向计算
在 CAPI 中，通过调用 `paddle_gradient_machine_forward` 接口，我们可以实现神经网络的前向计算。示例代码如下：
```c
paddle_arguments out_args = paddle_arguments_create_none();
CHECK(paddle_gradient_machine_forward(machine,
                                      in_args,
                                      out_args,
                                      /* isTrain */ false));
```
上述代码中的 `out_args` 保存了神经网络的输出结果。

### 5. 清理
在得到模型的预测结果之后，我们需要对使用的中间变量进行清理，以防内存泄露等问题的出现。示例代码如下：
```c
CHECK(paddle_matrix_destroy(prob));
CHECK(paddle_arguments_destroy(out_args));
CHECK(paddle_matrix_destroy(mat));
CHECK(paddle_arguments_destroy(in_args));
CHECK(paddle_gradient_machine_destroy(machine));
```

## 3. Python 接口的输入数据类型，到 CAPI 接口输入数据类型对应

Python 接口输入数据类型  | CAPI 接口输入数据类型
:-------------: | :-------------:
integer_value  | XX
dense_vector  | XX
sparse_binary_vector | XX
sparse_vector | XX
integer_value_sequence  | XX
dense_vector_sequence  | XX
sparse_binary_vector_sequence | XX
sparse_vector_sequence | XX
integer_value_sub_sequence  | XX
dense_vector_sub_sequence  | XX
sparse_binary_vector_sub_sequence | XX
sparse_vector_sub_sequence | XX

## 4. 多线程预测
[TBD]

## 5. F&Q
[TBD]
