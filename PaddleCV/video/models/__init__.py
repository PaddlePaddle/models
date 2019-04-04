from .model import regist_model, get_model
from .attention_cluster import AttentionCluster
from .attention_lstm import AttentionLSTM
from .nextvlad import NEXTVLAD
from .nonlocal_model import NonLocal
from .tsm import TSM
from .tsn import TSN
from .stnet import STNET

# regist models, sort by alphabet
regist_model("AttentionCluster", AttentionCluster)
regist_model("AttentionLSTM", AttentionLSTM)
regist_model("NEXTVLAD", NEXTVLAD)
regist_model('NONLOCAL', NonLocal)
regist_model("TSM", TSM)
regist_model("TSN", TSN)
regist_model("STNET", STNET)
