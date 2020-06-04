from .reader_utils import regist_reader, get_reader
from .kinetics_reader import KineticsReader

# regist reader, sort by alphabet
regist_reader("TSN", KineticsReader)
