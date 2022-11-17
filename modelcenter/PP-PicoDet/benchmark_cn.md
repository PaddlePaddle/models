## Benchmark
| 模型     | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | 参数量<br><sup>(M) | FLOPS<br><sup>(G) | 预测时延<sup><small>[CPU](#latency)</small><sup><br><sup>(ms) | 预测时延<sup><small>[Lite](#latency)</small><sup><br><sup>(ms) |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| PicoDet-XS |  320*320   |          23.5           |        36.1       |        0.70        |       0.67        |              3.9ms              |            7.81ms             |
| PicoDet-XS |  416*416   |          26.2           |        39.3        |        0.70        |       1.13        |              6.1ms             |            12.38ms             | 
| PicoDet-S |  320*320   |          29.1           |        43.4        |        1.18       |       0.97       |             4.8ms              |            9.56ms             | 
| PicoDet-S |  416*416   |          32.5           |        47.6        |        1.18        |       1.65       |              6.6ms              |            15.20ms             |
| PicoDet-M |  320*320   |          34.4           |        50.0        |        3.46        |       2.57       |             8.2ms              |            17.68ms             |
| PicoDet-M |  416*416   |          37.5           |        53.4       |        3.46        |       4.34        |              12.7ms              |            28.39ms            | 
| PicoDet-L |  320*320   |          36.1           |        52.0        |        5.80       |       4.20        |              11.5ms             |            25.21ms           |
| PicoDet-L |  416*416   |          39.4           |        55.7        |        5.80        |       7.10       |              20.7ms              |            42.23ms            | 
| PicoDet-L |  640*640   |          42.6           |        59.2        |        5.80        |       16.81        |              62.5ms              |            108.1ms          |

**注意:**
- 时延测试：</a> 我们所有的模型都在`英特尔酷睿i7 10750H`的CPU 和`骁龙865(4xA77+4xA55)`的ARM CPU上测试(4线程，FP16预测)。上面表格中标有`CPU`的是使用OpenVINO测试，标有`Lite`的是使用[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)进行测试。
- PicoDet在COCO train2017上训练，并且在COCO val2017上进行验证。使用4卡GPU训练，并且上表所有的预训练模型都是通过发布的默认配置训练得到。
- Benchmark测试：测试速度benchmark性能时，导出模型后处理不包含在网络中，需要设置`-o export.benchmark=True` 或手动修改[runtime.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/runtime.yml#L12)。


