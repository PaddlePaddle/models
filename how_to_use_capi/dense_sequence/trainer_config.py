from paddle.trainer_config_helpers import *

feat_dim = 5

sentence = data_layer(name='sentence', size=feat_dim)
lstm = simple_lstm(input=sentence, size=64)
lstm_last = last_seq(input=lstm)
outputs(fc_layer(input=lstm_last, size=2, act=SoftmaxActivation()))
