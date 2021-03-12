import config as c
import paddle.nn as nn
import paddle

from paddle import ParamAttr
import paddle
import paddle as pd 
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2D(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias_attr=False)
                              
        self.conv2 = nn.Conv2D(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias_attr=False)
                              
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14(nn.Layer):
    def __init__(self, sample_rate, window_size,
                 hop_size, mel_bins, fmin, 
        fmax, classes_num,extract_embedding=False):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.bn0 = nn.BatchNorm2D(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias_attr=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias_attr=True)
        self.sigmoid = nn.Sigmoid()
        self.extract_embedding = True
 
    def forward(self, x, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        #x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        #x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose([0,3,2,1])
        x = self.bn0(x)
        x = x.transpose([0,3,2,1])
        
     #   if self.training:
          #  x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        #x = torch.mean(x, dim=3)
        x = x.mean(axis=3,keepdim=True)
        
        #(x1, _) = torch.max(x, dim=2)
        x1 = x.max(axis=2,keepdim=True)
        #x2 = torch.mean(x, dim=2)
        x2 = x1.mean(axis=2,keepdim=True)
        
        x = x1 + x2
        x = x.squeeze()
        x = x.unsqueeze(0)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        
        embedding = F.dropout(x, p=0.5, training=self.training)
        if not self.extract_embedding:
            clipwise_output = self.sigmoid(self.fc_audioset(x))
            output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        else:
            output_dict = {'embedding': embedding}
            

        return output_dict
    

    
    
    
class ESCModel(nn.Layer):
    def __init__(self,pretrained = True):
        super(ESCModel, self).__init__()
        
        self.audioset_model = Cnn14(sample_rate=c.sample_rate, 
              window_size=c.window_size,
        hop_size=c.hop_size, mel_bins=c.mel_bins,
              fmin=c.fmin, fmax=c.fmax, 
            classes_num=527,extract_embedding=True)
        if pretrained:
            sd = paddle.load(c.audioset_checkpoint)
            self.audioset_model.load_dict(sd)
            print('pretrained model loaded from',c.audioset_checkpoint)
        self.fc_esc50 = nn.Linear(2048, 50, bias_attr=True)
        self.softmax = nn.Softmax()
        #elf.drop = nn.Dropout(c.dropout)
    def forward(self,X):
 
        out = self.audioset_model(X)
       # out = self.drop(out['embedding'])
        logits = self.fc_esc50(out['embedding'])
        return logits#self.softmax(logits)
        
        

#         #print(p)
