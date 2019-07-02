## ActivityNet 指标计算

- ActivityNet数据集的具体使用说明可以参考其[官方网站](http://activity-net.org)

- 下载指标评估代码，请从[ActivityNet Gitub repository](https://github.com/activitynet/ActivityNet.git)下载

- 计算精度指标

    ```cd ActivityNet/Evaluation```

    ```python get_detection_performance.py ./data/activity_net.v1-3.min.json $Test_Result```

  其中Test_Result是运行测试程序test.py输出的json格式的文件。
