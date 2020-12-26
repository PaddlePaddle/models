import os
import torch
#import paddle
import argparse
import numpy as np
from tqdm import tqdm

original_model_dir = "./CPM-large/80000"
# 加载原始模型
m0 = torch.load(
    os.path.join(original_model_dir, 'mp_rank_00_model_states.pt'),
    map_location='cpu')
m1 = torch.load(
    os.path.join(original_model_dir, 'mp_rank_01_model_states.pt'),
    map_location='cpu')

# 模型参数转换
state_dict = {}
for x, y in tqdm(zip(m0['module'].items(), m1['module'].items())):
    name_0, param_0 = x
    name_1, param_1 = y
    print(name_0, name_1)
