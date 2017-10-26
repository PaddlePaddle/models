import os
from collections import defaultdict


def get_file_list(image_file_list):
    '''
    Generate the file list for training and testing data.
    
    :param image_file_list: The path of the file which contains
                            path list of image files.
    :type image_file_list: str
    '''
    dirname = os.path.dirname(image_file_list)
    path_list = []
    with open(image_file_list) as f:
        for line in f:
            line_split = line.strip().split(',', 1)
            filename = line_split[0].strip()
            path = os.path.join(dirname, filename)
            label = line_split[1][2:-1].strip()
            if label:
                path_list.append((path, label))

    return path_list


def build_label_dict(file_list, save_path):
    """
    Build label dictionary from training data.
    
    :param file_list: The list which contains the labels 
                      of training data.
    :type file_list: list
    :params save_path: The path where the label dictionary will be saved.
    :type save_path: str
    """
    values = defaultdict(int)
    for path, label in file_list:
        for c in label:
            if c:
                values[c] += 1

    values['<unk>'] = 0
    with open(save_path, "w") as f:
        for v, count in sorted(
                values.iteritems(), key=lambda x: x[1], reverse=True):
            f.write("%s\t%d\n" % (v, count))


def load_dict(dict_path):
    """
    Load label dictionary from the dictionary path.
    
    :param dict_path: The path of word dictionary.
    :type dict_path: str
    """
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    """
    Load the reversed label dictionary from dictionary path.
    
    :param dict_path: The path of word dictionary.
    :type dict_path: str
    """
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
