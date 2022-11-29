# reprod_log

主要用于对比和记录模型复现过程中的各个步骤精度对齐情况
## 安装

1. 本地编译安装
```bash
cd models/tutorials/reprod_log
python3 setup.py bdist_wheel
python3 install dist/reprod_log-x.x.-py3-none-any.whl --force-reinstall
```

2. pip直接安装
```bash
# from pypi
pip3 install reprod_log --force-reinstall
# from bcebos
pip3 install https://paddle-model-ecology.bj.bcebos.com/whl/reprod_log-1.0.1-py3-none-any.whl
```
## 提供的类和方法

### 论文复现赛

在论文复现赛中，主要用到的类如下所示。

* ReprodLogger
    * 功能：记录和保存复现过程中的中间变量，用于后续的diff排查
    * 初始化参数：无
    * 方法
        * add(key, val)
            * 功能：向logger中添加key-val pair
            * 输入
                * key (str) : PaddlePaddle中的key与参考代码中保存的key应该完全相同，否则会提示报错
                * value (numpy.ndarray) : key对应的值
            * 返回: None
        * remove(key)
            * 功能：移除logger中的关键字段key及其value
            * 输入
                * key (str) : 关键字段
                * value (numpy.ndarray) : key对应的值
            * 返回: None
        * clear()
            * 功能：清空logger中的关键字段key及其value
            * 输入: None
            * 返回: None
        * save(path)
            * 功能：将logger中的所有的key-value信息保存到文件中
            * 输入:
                * path (str): 路径
            * 返回: None
* ReprodDiffHelper
    * 功能：对`ReprodLogger`保存的日志文件进行解析，打印与记录diff
    * 初始化参数：无
    * 方法
        * load_info(path)
            * 功能：加载
            * 输入:
                * path (str): 日志文件路径
            * 返回: dict信息，key为str，value为numpy.ndarray
        * compare_info(info1, info2)
            * 功能：计算两个字典对于相同key的value的diff，具体计算方法为`diff = np.abs(info1[key] - info2[key])`
            * 输入:
                * info1/info2 (dict): PaddlePaddle与参考代码保存的文件信息
            * 返回: diff的dict信息
        * report(diff_method="mean", diff_threshold=1e-6, path="./diff.txt")
            * 功能：可视化diff，保存到文件或者到屏幕
            * 参数
                * diff_method (str): diff计算方法，包括`mean`、`min`、`max`、`all`，默认为`mean`
                * diff_threshold (float): 阈值，如果diff大于该阈值，则核验失败，默认为`1e-6`
                * path (str): 日志保存的路径，默认为`./diff.txt`




### more

类 `ReprodLogger` 用于记录和报错复现过程中的中间变量

主要方法为

* add(key, val)：添加key-val pair
* remove(key)：移除key
* clear()：清空字典
* save(path)：保存字典

类 `ReprodDiffHelper` 用于对中间变量进行检查，主要为计算diff

主要方法为

* load_info(path): 加载字典文件
* compare_info(info1:dict, info2:dict): 对比diff
* report(diff_threshold=1e-6,path=None): 可视化diff，保存到文件或者到屏幕

模块 `compare` 提供了基础的网络前向和反向过程对比工具

* compare_forward 用于对比网络的前向过程，其参数为
  * torch_model: torch.nn.Module,
  * paddle_model: paddle.nn.Layer,
  * input_dict: dict, dict值为numpy矩阵
  * diff_threshold: float=1e-6
  * diff_method: str = 'mean' 检查diff的函数，目前支持 min,max,mean,all四种形式，并且支持min,max,mean的相互组合成的list形式，如['min','max']

* compare_loss_and_backward 用于对比网络的反向过程，其参数为
  * torch_model: torch.nn.Module,
  * paddle_model: paddle.nn.Layer,
  * torch_loss: torch.nn.Module,
  * paddle_loss: paddle.nn.Layer,
  * input_dict: dict, dict值为numpy矩阵
  * lr: float=1e-3,
  * steps: int=10,
  * diff_threshold: float=1e-6
  * diff_method: str = 'mean' 检查diff的函数，目前支持 min,max,mean,all四种形式，并且支持min,max,mean的相互组合成的list形式，如['min','max']
