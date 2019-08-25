scripts：运行数据处理脚本目录, 将官方公开数据集转换成模型所需训练数据格式
运行命令：
  sh run_build_data.sh [udc|swda|mrda|atis|dstc2]

1)、生成MATCHING任务所需要的训练集、开发集、测试集时:
sh run_build_data.sh udc
生成数据在dialogue_general_understanding/data/input/data/udc

2)、生成DA任务所需要的训练集、开发集、测试集时: 
  sh run_build_data.sh swda
  sh run_build_data.sh mrda
  生成数据分别在dialogue_general_understanding/data/input/data/swda和dialogue_general_understanding/data/input/data/mrda

3)、生成DST任务所需的训练集、开发集、测试集时:
  sh run_build_data.sh dstc2
  生成数据分别在dialogue_general_understanding/data/input/data/dstc2

4)、生成意图解析, 槽位识别任务所需训练集、开发集、测试集时:
  sh run_build_data.sh atis
  生成槽位识别数据在dialogue_general_understanding/data/input/data/atis/atis_slot
  生成意图识别数据在dialogue_general_understanding/data/input/data/atis/atis_intent



