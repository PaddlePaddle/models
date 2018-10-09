# Text matching on Quora qestion-answer pair dataset

## Environment Preparation

### install python2

TODO

### Install fluid 0.15.0

TODO

## Prepare Data

Please download the Quora dataset firstly from https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing
 to ROOT_DIR $HOME/.cache/paddle/dataset

Then run the data/prepare_quora_data.sh to download the pretrained embedding glove.840B.300d.zip:

```shell
cd data
sh prepare_quora_data.sh   
```

The finally dataset dir should be like

```shell

$HOME/.cache/paddle/dataset
    |- Quora_question_pair_partition
        |- train.tsv
        |- test.tsv
        |- dev.tsv
        |- readme.txt
        |- wordvec.txt
    |- glove.840B.300d.txt
```


### Train

```shell
fluid train_and_evaluate.py  \
    --model_name=cdssmNet  \
    --config=cdssm_base
```

You are supposed to get log like this:

```shell
net_name:  cdssmNet
config {'save_dirname': 'cdssm_model', 'optimizer_type': 'adam', 'duplicate_data': False, 'train_samples_num': 384348, 'droprate_fc': 0.1, 'fc_dim': 128, 'kernel_count': 300, 'mlp_hid_dim': [128, 128], 'OOV_fill': 'uniform', 'class_dim': 2, 'epoch_num': 50, 'lr_decay': 1, 'learning_rate': 0.001, 'batch_size': 128, 'use_lod_tensor': True, 'metric_type': ['accuracy'], 'embedding_norm': False, 'emb_dim': 300, 'droprate_conv': 0.1, 'use_pretrained_word_embedding': True, 'kernel_size': 5, 'dict_dim': 40000}
Generating word dict...
('Vocab size: ', 36057)
loading word2vec from  data/glove.840B.300d.txt
preparing pretrained word embedding ...
pretrained_word_embedding to be load: [[-0.086864    0.19161     0.10915    ... -0.01516     0.11108
   0.2065    ]
 [ 0.27204    -0.06203    -0.1884     ...  0.13015    -0.18317
   0.1323    ]
 [-0.20628     0.36716    -0.071933   ...  0.14271     0.50059
   0.038025  ]
 ...
 [-0.0387745   0.03030911 -0.01028247 ... -0.03096982 -0.01002833
   0.04407753]
 [-0.02707165 -0.04616793 -0.0260934  ... -0.00642176  0.02934359
   0.02570623]
 [ 0.00578131  0.0343625  -0.02623712 ... -0.04737288  0.01997969
   0.04304557]]
param name: emb.w; param shape: (40000L, 300L)
param name: conv1d.w; param shape: (1500L, 300L)
param name: fc1.w; param shape: (300L, 128L)
param name: fc1.b; param shape: (128L,)
param name: fc_2.w_0; param shape: (256L, 128L)
param name: fc_2.b_0; param shape: (128L,)
param name: fc_3.w_0; param shape: (128L, 128L)
param name: fc_3.b_0; param shape: (128L,)
param name: fc_4.w_0; param shape: (128L, 2L)
param name: fc_4.b_0; param shape: (2L,)
loading pretrained word embedding to param
[Tue Oct  9 12:48:35 2018] epoch_id: -1, dev_cost: 0.796980, accuracy: 0.5
[Tue Oct  9 12:48:36 2018] epoch_id: -1, test_cost: 0.796876, accuracy: 0.5

[Tue Oct  9 12:48:36 2018] Start Training
[Tue Oct  9 12:48:44 2018] epoch_id: 0, batch_id: 0, cost: 0.878309, acc: 0.398438
[Tue Oct  9 12:48:46 2018] epoch_id: 0, batch_id: 100, cost: 0.607255, acc: 0.664062
[Tue Oct  9 12:48:48 2018] epoch_id: 0, batch_id: 200, cost: 0.521560, acc: 0.765625
[Tue Oct  9 12:48:51 2018] epoch_id: 0, batch_id: 300, cost: 0.512380, acc: 0.734375
[Tue Oct  9 12:48:54 2018] epoch_id: 0, batch_id: 400, cost: 0.522022, acc: 0.703125
[Tue Oct  9 12:48:56 2018] epoch_id: 0, batch_id: 500, cost: 0.470358, acc: 0.789062
[Tue Oct  9 12:48:58 2018] epoch_id: 0, batch_id: 600, cost: 0.561773, acc: 0.695312
[Tue Oct  9 12:49:01 2018] epoch_id: 0, batch_id: 700, cost: 0.485580, acc: 0.742188
[Tue Oct  9 12:49:03 2018] epoch_id: 0, batch_id: 800, cost: 0.493103, acc: 0.765625
[Tue Oct  9 12:49:05 2018] epoch_id: 0, batch_id: 900, cost: 0.388173, acc: 0.804688
[Tue Oct  9 12:49:08 2018] epoch_id: 0, batch_id: 1000, cost: 0.511332, acc: 0.742188
[Tue Oct  9 12:49:10 2018] epoch_id: 0, batch_id: 1100, cost: 0.488231, acc: 0.734375
[Tue Oct  9 12:49:12 2018] epoch_id: 0, batch_id: 1200, cost: 0.438371, acc: 0.781250
[Tue Oct  9 12:49:15 2018] epoch_id: 0, batch_id: 1300, cost: 0.535350, acc: 0.750000
[Tue Oct  9 12:49:17 2018] epoch_id: 0, batch_id: 1400, cost: 0.459860, acc: 0.773438
[Tue Oct  9 12:49:19 2018] epoch_id: 0, batch_id: 1500, cost: 0.382312, acc: 0.796875
[Tue Oct  9 12:49:22 2018] epoch_id: 0, batch_id: 1600, cost: 0.480827, acc: 0.742188
[Tue Oct  9 12:49:24 2018] epoch_id: 0, batch_id: 1700, cost: 0.474005, acc: 0.789062
[Tue Oct  9 12:49:26 2018] epoch_id: 0, batch_id: 1800, cost: 0.421068, acc: 0.789062
[Tue Oct  9 12:49:28 2018] epoch_id: 0, batch_id: 1900, cost: 0.420553, acc: 0.789062
[Tue Oct  9 12:49:31 2018] epoch_id: 0, batch_id: 2000, cost: 0.458412, acc: 0.781250
[Tue Oct  9 12:49:33 2018] epoch_id: 0, batch_id: 2100, cost: 0.360774, acc: 0.859375
[Tue Oct  9 12:49:35 2018] epoch_id: 0, batch_id: 2200, cost: 0.361226, acc: 0.835938
[Tue Oct  9 12:49:37 2018] epoch_id: 0, batch_id: 2300, cost: 0.371504, acc: 0.843750
[Tue Oct  9 12:49:40 2018] epoch_id: 0, batch_id: 2400, cost: 0.449930, acc: 0.804688
[Tue Oct  9 12:49:42 2018] epoch_id: 0, batch_id: 2500, cost: 0.442774, acc: 0.828125
[Tue Oct  9 12:49:44 2018] epoch_id: 0, batch_id: 2600, cost: 0.471352, acc: 0.781250
[Tue Oct  9 12:49:46 2018] epoch_id: 0, batch_id: 2700, cost: 0.344527, acc: 0.875000
[Tue Oct  9 12:49:48 2018] epoch_id: 0, batch_id: 2800, cost: 0.450750, acc: 0.765625
[Tue Oct  9 12:49:51 2018] epoch_id: 0, batch_id: 2900, cost: 0.459296, acc: 0.835938
[Tue Oct  9 12:49:53 2018] epoch_id: 0, batch_id: 3000, cost: 0.495118, acc: 0.742188

[Tue Oct  9 12:49:53 2018] epoch_id: 0, train_avg_cost: 0.457090, train_avg_acc: 0.779325
[Tue Oct  9 12:49:54 2018] epoch_id: 0, dev_cost: 0.439462, accuracy: 0.7865
[Tue Oct  9 12:49:55 2018] epoch_id: 0, test_cost: 0.441658, accuracy: 0.7867

[Tue Oct  9 12:50:04 2018] epoch_id: 1, batch_id: 0, cost: 0.320335, acc: 0.843750
[Tue Oct  9 12:50:06 2018] epoch_id: 1, batch_id: 100, cost: 0.398587, acc: 0.820312
[Tue Oct  9 12:50:08 2018] epoch_id: 1, batch_id: 200, cost: 0.324227, acc: 0.843750
[Tue Oct  9 12:50:11 2018] epoch_id: 1, batch_id: 300, cost: 0.303423, acc: 0.890625
[Tue Oct  9 12:50:13 2018] epoch_id: 1, batch_id: 400, cost: 0.438270, acc: 0.812500
[Tue Oct  9 12:50:15 2018] epoch_id: 1, batch_id: 500, cost: 0.307846, acc: 0.828125
[Tue Oct  9 12:50:19 2018] epoch_id: 1, batch_id: 600, cost: 0.338888, acc: 0.851562
[Tue Oct  9 12:50:21 2018] epoch_id: 1, batch_id: 700, cost: 0.341852, acc: 0.843750
[Tue Oct  9 12:50:23 2018] epoch_id: 1, batch_id: 800, cost: 0.365191, acc: 0.820312
[Tue Oct  9 12:50:25 2018] epoch_id: 1, batch_id: 900, cost: 0.464820, acc: 0.804688
[Tue Oct  9 12:50:28 2018] epoch_id: 1, batch_id: 1000, cost: 0.348680, acc: 0.851562
[Tue Oct  9 12:50:30 2018] epoch_id: 1, batch_id: 1100, cost: 0.390921, acc: 0.828125
[Tue Oct  9 12:50:32 2018] epoch_id: 1, batch_id: 1200, cost: 0.361488, acc: 0.820312
[Tue Oct  9 12:50:35 2018] epoch_id: 1, batch_id: 1300, cost: 0.324751, acc: 0.851562
[Tue Oct  9 12:50:37 2018] epoch_id: 1, batch_id: 1400, cost: 0.428706, acc: 0.804688
[Tue Oct  9 12:50:39 2018] epoch_id: 1, batch_id: 1500, cost: 0.504243, acc: 0.742188
[Tue Oct  9 12:50:42 2018] epoch_id: 1, batch_id: 1600, cost: 0.322159, acc: 0.851562
[Tue Oct  9 12:50:44 2018] epoch_id: 1, batch_id: 1700, cost: 0.451969, acc: 0.757812
[Tue Oct  9 12:50:46 2018] epoch_id: 1, batch_id: 1800, cost: 0.298705, acc: 0.890625
[Tue Oct  9 12:50:49 2018] epoch_id: 1, batch_id: 1900, cost: 0.439283, acc: 0.789062
[Tue Oct  9 12:50:51 2018] epoch_id: 1, batch_id: 2000, cost: 0.325409, acc: 0.851562
[Tue Oct  9 12:50:53 2018] epoch_id: 1, batch_id: 2100, cost: 0.312230, acc: 0.875000
[Tue Oct  9 12:50:56 2018] epoch_id: 1, batch_id: 2200, cost: 0.352170, acc: 0.843750
[Tue Oct  9 12:50:58 2018] epoch_id: 1, batch_id: 2300, cost: 0.366158, acc: 0.828125
[Tue Oct  9 12:51:00 2018] epoch_id: 1, batch_id: 2400, cost: 0.349191, acc: 0.812500
[Tue Oct  9 12:51:02 2018] epoch_id: 1, batch_id: 2500, cost: 0.391564, acc: 0.835938
[Tue Oct  9 12:51:05 2018] epoch_id: 1, batch_id: 2600, cost: 0.347518, acc: 0.835938
[Tue Oct  9 12:51:07 2018] epoch_id: 1, batch_id: 2700, cost: 0.279777, acc: 0.914062
[Tue Oct  9 12:51:09 2018] epoch_id: 1, batch_id: 2800, cost: 0.293878, acc: 0.851562
[Tue Oct  9 12:51:11 2018] epoch_id: 1, batch_id: 2900, cost: 0.367596, acc: 0.843750
[Tue Oct  9 12:51:13 2018] epoch_id: 1, batch_id: 3000, cost: 0.433259, acc: 0.804688

[Tue Oct  9 12:51:14 2018] epoch_id: 1, train_avg_cost: 0.348265, train_avg_acc: 0.841591
[Tue Oct  9 12:51:15 2018] epoch_id: 1, dev_cost: 0.398465, accuracy: 0.8163
[Tue Oct  9 12:51:16 2018] epoch_id: 1, test_cost: 0.399254, accuracy: 0.8209
```
