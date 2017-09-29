## 使用C-API实现nest sequence类型数据的预测

### 1. 将网络配置转为protobuf二进制文件
使用C-API进行预测时，需要先将网络配置转换为protobuf二进制文件，在本例的文件目录中包含 ```convert_protobin.sh``` , 它通过调用python中的```dump_config```模块来将目录下的配置文件 ```trainer_config.conf```转化为```trainer_config.bin```。 执行以下命令完成转换：
```shell
sh convert_protobin.sh
```

### 2. 输入数据
本示例的输入数据格式为`paddle.data_type.integer_value_sub_sequence`，即每个样本是由双层序列构成。以样例数据data.txt为例：
```
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```
数据的每个元素对应的是词表ID。

#### 序列数据输入
C-API提供了```paddle_arguments_set_sequence_start_pos```接口来标记每个样本里序列的起始位置，示例代码如下：

```c
int seq_pos_array[] = {0, 10, 20};
paddle_ivector seq_pos = paddle_ivector_create(
    seq_pos_array, sizeof(seq_pos_array) / sizeof(int), false, false);
CHECK(paddle_arguments_set_sequence_start_pos(in_args, 0, 0, seq_pos));
```

其中，``` seq_pos_arry ``` 的前两个元素表示对应位置的样本中序列开始的位置，而最后一个元素表示输入数据中所有的单词数量。通过调用```paddle_arguments_set_sequence_start_pos```接口，示例代码将输入数据转化为2个序列样本数据:

```
[ [1 2 3 4 5 6 7 8 9 10],
  [11 12 13 14 15 16 17 18 19 20] ]
```

#### sub-sequence 数据
当输入数据包含双层序列时，样本的每个元素是单层序列，称之为双层序列的一个子序列(sub-sequence), sub-sequence 的每个元素是词表ID。```paddle_arguments_set_sequence_start_pos```接口同样可以标记每个样本里 sub-sequence 的起始位置， 示例代码如下：

```c
int sub_seq_pos_array[] = {0, 5, 10, 15, 20};
paddle_ivector sub_seq_pos = paddle_ivector_create(
    sub_seq_pos_array, sizeof(sub_seq_pos_array) / sizeof(int), false, false);
CHECK(paddle_arguments_set_sequence_start_pos(in_args, 0, 1, sub_seq_pos));
```

其中，``` sub_seq_pos_array ``` 的前四个元素表示对应位置的 sub-sequence 中序列开始的位置，而最后一个元素表示输入数据中所有的单词数量。```paddle_arguments_set_sequence_start_pos``` 的第三个参数表示nest_level, 当设置 sub-sequence 序列的起始位置时， 需要将其设置为1。 通过调用```paddle_arguments_set_sequence_start_pos```接口，示例代码将输入数据转化为两个双层序列的样本：

```
[ [[1 2 3 4 5],
   [6 7 8 9 10]],

  [[11 12 13 14 15],
  [16 17 18 19 20]] ]
```


### 3. 模型预测
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
./nest_sequence
```
