python infer.py --use_gpu 0 \  #是否使用GPU
--batch_size 64 \  #batch_size大小
--model_dir "./model_dir" \  #模型保存路径
--TRIGRAM_D 1000 \  #trigram后的向量维度
--L1_N 300 \  #第一层mlp大小
--L2_N 300 \  #第二层mlp大小
--L3_N 128 \  #第三层mlp大小
--Neg 4 \  #负样本采样数量
