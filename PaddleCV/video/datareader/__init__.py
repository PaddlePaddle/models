from .reader_utils import regist_reader, get_reader
from .feature_reader import FeatureReader
from .kinetics_reader import KineticsReader
from .nonlocal_reader import NonlocalReader

# regist reader, sort by alphabet
regist_reader("ATTENTIONCLUSTER", FeatureReader)
regist_reader("ATTENTIONLSTM", FeatureReader)
regist_reader("NEXTVLAD", FeatureReader)
regist_reader("NONLOCAL", NonlocalReader)
regist_reader("TSM", KineticsReader)
regist_reader("TSN", KineticsReader)
regist_reader("STNET", KineticsReader)
