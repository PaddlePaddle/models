### 使用C-API实现dense vector sequence类型数据的预测

#### 1. 将网络配置转为protobuf二进制文件
使用C-API进行预测时，需要先将网络配置转换为protobuf二进制文件，在本例的文件目录中包含 ```convert_protobin.sh``` , 它通过调用python中的```dump_config```模块来将目录下的配置文件 ```trainer_config.conf```转化为```trainer_config.bin```。 执行以下命令完成转换：
```shell
sh convert_protobin.sh
```

#### 2. 输入数据
本示例的输入数据格式为dense vector sequece，即每个样本是由dense vector构成的序列。以样例数据data.txt为例：
```
1 2 3 4 5
6 7 8 9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
26 27 28 29 30
```
数据的每行表示一个长度为5的特征向量，数据格式为dense vector。

C-API提供了```paddle_arguments_set_sequence_start_pos```接口来标记每个样本里序列的起始位置，示例代码如下：
```c
int seq_pos_array[] = {0, 1, 3, 6};
paddle_ivector seq_pos = paddle_ivector_create(
    seq_pos_array, sizeof(seq_pos_array) / sizeof(int), false, false);
CHECK(paddle_arguments_set_sequence_start_pos(in_args, 0, 0, seq_pos));
```

其中，``` seq_pos_arry ``` 的前3个元素表示对应位置的样本中序列开始的位置，而最后一个元素表示输入数据中所有的dense vector的数量。通过调用```paddle_arguments_set_sequence_start_pos```接口，示例代码将输入数据转化为3个dense vector sequece类型的样本数据:

```
[[[1 2 3 4 5]],

[[6 7 8 9 10],
[11 12 13 14 15]],

[[16 17 18 19 20],
[21 22 23 24 25],
[26 27 28 29 30]]]
```

#### 3. 模型预测
1. 本示例运行需要依赖链接库```libpaddle_capi_shared.so```和C-API包含的头文件（```capi.h```等），在编译程序之前请将其所在的目录分别加入到系统的动态链接库和头文件的环境变量中。

2. 在当前目录将网络配置转为protobuf二进制文件
```
sh convert_protobin.sh
```
3. 创建编译目录
```
mkdir build && cd build
```
4. 编译执行
```
cmake ..
make
./dense_sequence
```
