from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import distutils.util
import numpy as np
from paddle.fluid import core
import six 
import cv2

def check_data_valid(data_path):
    """
    check if dataset exists
    """
    if not os.path.exists(data_path):
        return False
    return True


def check_save_valid(checkpoint_path):
    """
    if not checkpoint_path exists, creat it
    """
    
    if not os.path.exists(checkpoint_path):
        try:
            os.mkdir(checkpoint_path)
        except:
            return False
    return True


def read_data(data_path, batch_size, ext, scale_factor, bia_size):
    def data_reader():
        img_inputs = []
        img_gths = []
        count_bt = 0
        for image in os.listdir(data_path):
            if image.endswith(ext):
                img = cv2.imread(os.path.join(data_path, image))
                yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                img_y, img_u, img_v = cv2.split(yuv)
                j = 0
                while j+33 < len(img_y):
                    i = 0
                    while i+33 < len(img_y[0]):
                        img_patch = img_y[j:j+33, i:i+33]
                        img_gth = img_patch[bia_size:33-bia_size, bia_size:33-bia_size].copy()
                        img_blur = cv2.GaussianBlur(img_patch, (5, 5), 0)
                        img_sumsample = cv2.resize(img_blur, (int(33/scale_factor), int(33/scale_factor)))
                        img_input = cv2.resize(img_blur, (33, 33), interpolation=cv2.INTER_CUBIC)
                        img_inputs.append([img_input])
                        img_gths.append([img_gth])
                        count_bt += 1
                        if count_bt % batch_size == 0:
                            yield [[np.array(img_inputs), np.array(img_gths)]]
                            img_inputs = []
                            img_gths = []                            
                        i+=14
                    j+= 14
    return data_reader


class SRCNNStruct(object):
    def __init__(self, f1, f2, f3, n1, n2):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.n1 = n1
        self.n2 = n2


def print_arguments(args):
    """Print argparse's arguments.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)
    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
**kwargs)

