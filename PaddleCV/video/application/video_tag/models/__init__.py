from .model import regist_model, get_model
from .attention_lstm import AttentionLSTM
from .tsn import TSN

# regist models, sort by alphabet
regist_model("AttentionLSTM", AttentionLSTM)
regist_model("TSN", TSN)
