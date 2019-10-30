## ActivityNet 指标计算


- ActivityNet数据集的具体使用说明可以参考其[官方网站](http://activity-net.org)

- 下载指标评估代码，请从[ActivityNet Gitub repository](https://github.com/activitynet/ActivityNet.git)下载，将Evaluation文件夹拷贝至PaddleVideo目录下。(注：若使用python3，print函数需要添加括号，请对Evaluation目录下的.py文件做相应修改。)

- 请下载[activity\_net\_1\_3\_new.json](https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json)文件，并将其放置在PaddleVideo/Evaluation/data目录下，相较于原始的activity\_net.v1-3.min.json文件，我们过滤了其中一些失效的视频条目。

- 计算精度指标

    ```cd metrics/bmn_metrics```

    ```python eval_anet_prop.py```
