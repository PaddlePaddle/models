scripts：运行数据处理脚本目录
运行命令：
  sh run_build_data.sh [udc|swda|mrda|atis]

生成DA任务所需要的训练集、开发集、测试集时: 
  sh run_build_data.sh swda
  sh run_build_data.sh mrda
  生成数据分别在open-dialog/data/swda和open-dialog/data/mrda

生成DST任务所需的训练集、开发集、测试集时:
  sh run_build_data.sh dstc2
  生成数据分别在open-dialog/data/dstc2

生成意图解析, 槽位识别任务所需训练集、开发集、测试集时:
  sh run_build_data.sh atis
  生成槽位识别数据在open-dialog/data/atis/atis_slot
  生成意图识别数据在open-dialog/data/atis/atis_intent



