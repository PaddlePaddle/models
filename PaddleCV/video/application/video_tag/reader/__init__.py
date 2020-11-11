from .reader_utils import regist_reader, get_reader
from .feature_reader import FeatureReader
from .kinetics_reader import KineticsReader

# regist reader, sort by alphabet
regist_reader("ATTENTIONLSTM", FeatureReader)
regist_reader("TSN", KineticsReader)
