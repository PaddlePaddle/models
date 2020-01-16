# gru4rec 动态图实现

# 下载数据
```
wget https://paddlerec.bj.bcebos.com/gru4rec/dy_graph/data_rsc15.tar
tar xvf data_rsc15.tar
```

# 训练及预测

```
CUDA_VISIBLE_DEVICES=0 nohup sh run_gru.sh > log 2>&1 &
```

每一轮训练完都会进行预测。
