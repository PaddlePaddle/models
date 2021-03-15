# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataset_esc50 import get_loaders
from model_esc50 import ESCModel
import paddle
import argparse
import config as c
from visualdl import LogWriter

from train_utils import train_one_epoch,evaluate,test

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train esc50 by paddle')

    parser.add_argument('--test_fold', type=int, required=False,default=1)
    parser.add_argument('--device', type=int, required=False,default=0)

    last_model = None
    args = parser.parse_args([])
    paddle.set_device('gpu:{}'.format(args.device))
    logger = LogWriter(logdir = c.log_path)
    
    os.makedirs(c.model_path,exist_ok=True)
    
    model = ESCModel(pretrained=True)
    for p in model.parameters():
        p.stop_gradient = False
    optimizer = paddle.optimizer.Adam(learning_rate=3e-4, parameters=model.parameters())
    train_loader,val_loader,test_loader = get_loaders(args.test_fold)
    epoch_num = c.epoch_num
    loss_fn = paddle.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epoch_num):
        print('current lr {}'.format(optimizer.get_lr()))
        if epoch %20 == 0 and epoch !=0:
            optimizer.set_lr(optimizer.get_lr()*0.5)
        train_one_epoch(epoch,train_loader,model,optimizer,loss_fn,logger)
        #val_acc = evaluate(epoch,val_loader,model,loss_fn,logger)
        test_acc = test(epoch,test_loader,model,loss_fn,logger)

        if test_acc > best_acc:
            best_acc = test_acc
            if last_model is not None:
                os.remove(last_model)
            fn = os.path.join(c.model_path,'esc50_fold{}_epoch{}_acc_{:.3}.pd.tar'.format(args.test_fold,epoch,test_acc))
            paddle.save(model.state_dict(),fn)
            last_model = fn
            print('model saved to',fn)

#     train_loss_wrt.add_scalar(tag='train_loss', step=epoch, value=avg_loss_acc[0])
#     train_acc_wrt.add_scalar(tag='train_acc', step=epoch, value=avg_loss_acc[1])
#     val_loss_wrt.add_scalar(tag='val_loss', step=e
