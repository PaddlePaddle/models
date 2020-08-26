# extract_rawframes_opencv.py
## 应用说明

### 对于kinetics400数据
运行脚本的命令如下 `python extract_rawframes_opencv.py ./video/ ./rawframes/ --level 2 --ext mp4` 或者`python extract_rawframes_opencv.py ./video/ ./rawframes/ --level 2 --ext mp4`

### 参数说明
`./video/`       ： 这个参数表示视频目录的地址
`./rawframes`    ： 提取出的frames的存放目录
`--level 1 or 2` ：  

                        level 1，表示video的存储方式为

                                ------ video
                                        |------ xajhljklk.mp4
                                        |------ jjkjlljjk.mp4
                                        ....


                        level 2, 表示video的存储方式为
                        ------ video
                                |------ class1
                                        |-------- xajhljklk.mp4
                                        |-------- jjkjlljjk.mp4
                                ....
`--ext 4`        : 表示视频文件的格式。
