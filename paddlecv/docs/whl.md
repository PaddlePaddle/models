# Whl包使用

## 1. 安装与简介

目前该whl包尚未上传至pypi，因此目前需要通过以下方式安装。

```shell
python setup.py bdist_wheel
pip install dist/paddlecv-0.1.0-py3-none-any.whl
```

## 2. 基本调用


使用方式如下所示。

* 可以指定task_name或者config_path，来获取所需要预测的系统。当使用`task_name`时，会从PaddleCV项目中获取已经自带的模型或者串联系统，进行预测，而使用`config_path`时，则会加载配置文件，完成模型或者串联系统的初始化。
s
```py
from paddlecv import PaddleCV
paddlecv = PaddleCV(task_name="PP-OCRv3")
res = paddlecv("../demo/00056221.jpg")
```

* 如果希望查看系统自带的的串联系统列表，可以使用下面的方式。

```py
from paddlecv import PaddleCV
PaddleCV.list_all_supported_tasks()
```

输出内容如下。

```
[11/17 06:17:20] ppcv INFO: Tasks and recommanded configs that paddlecv supports are :
PP-Human: paddlecv://configs/system/PP-Human.yml
PP-OCRv2: paddlecv://configs/system/PP-OCRv2.yml
PP-OCRv3: paddlecv://configs/system/PP-OCRv3.yml
...
```


## 3. 高阶开发

如果你希望优化paddlecv whl包接口，可以修改`paddlecv.py`文件，然后重新编译生成whl包即可。
