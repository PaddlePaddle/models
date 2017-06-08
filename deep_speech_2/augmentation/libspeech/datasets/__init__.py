from . import base
from . import otherdatasets
from . import wsj

_DATASET_MAP = {
    'datasets.base.JSONDataset': base.JSONDataset,
    'datasets.net.JSONDataset': base.NetJSONDataset,
    'datasets.wsj.WSJSet': wsj.WSJSet,
    'datasets.otherdatasets.FisherSet': otherdatasets.FisherSet,
    'datasets.otherdatasets.SwbdSet': otherdatasets.SwbdSet
}


def get_dataset_class(class_str):
    """Return class (constructor) for dataset class named by `class_str`

    Example:
        >>> ClassHandle = get_dataset_class("datasets.base.JSONDataset")

    Args:
        class_str (str): String containing dataset class relative to
            libspeech root.

    Returns:
        class: class constructor
    """
    return _DATASET_MAP[class_str]
