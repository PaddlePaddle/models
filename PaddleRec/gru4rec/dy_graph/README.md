# gru4rec 动态图实现

# 环境配置
paddle 1.7



# 下载数据
```
wget https://paddlerec.bj.bcebos.com/gru4rec/dy_graph/data_rsc15.tar
tar xvf data_rsc15.tar
```

# 数据格式
数据格式及预处理处理同静态图相同。

# 训练及预测

```
CUDA_VISIBLE_DEVICES=0 nohup sh run_gru.sh > log 2>&1 &
```

每一轮训练完都会进行预测。
