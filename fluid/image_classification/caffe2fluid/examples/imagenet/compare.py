#!/usr/bin/python

#
#a tool to compare tensors in two files or two directories
#

import sys
import os


def walk_dir(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            yield file


def calc_diff(f1, f2):
    import numpy as np

    d1 = np.load(f1).flatten()
    d2 = np.load(f2).flatten()

    d1_num = reduce(lambda x, y: x * y, d1.shape)
    d2_num = reduce(lambda x, y: x * y, d2.shape)
    if d1_num != d2_num:
        print d1.shape
        print d2.shape
        assert (d1_num == d2_num), "their shape is not consistent"

    try:
        df = np.abs(d1 - d2)
        max_df = np.max(df)
        sq_df = np.mean(df * df)
        return max_df, sq_df
    except Exception as e:
        return -1.0, -1.0


def compare(path1, path2):
    def diff(f1, f2):
        max_df, sq_df = calc_diff(f1, f2)
        print('compare %s <=> %s with result[max_df:%.4e, sq_df:%.4e]' %
              (f1, f2, max_df, sq_df))
        assert (max_df < 1e-5), \
                'max_df is too large with value[%.6e]' % (max_df)
        assert (sq_df < 1e-10), \
                'sq_df is too large with value[%.6e]' % (sq_df)

    if os.path.exists(path1) is False:
        print('not found %s' % (path1))
        return 1
    elif os.path.exists(path2) is False:
        print('not found %s' % (path2))
        return 1

    if path1.find('.npy') > 0 and path2.find('.npy') > 0:
        diff(path1, path2)
        return

    for f in walk_dir(path2):
        if f.find('.npy') < 0:
            continue

        f1 = os.path.join(path1, f)
        f2 = os.path.join(path2, f)
        diff(f1, f2)

    print('all checking succeed to pass')
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        path1 = 'lenet.tf/results'
        path2 = 'lenet.paddle/results'
    elif len(sys.argv) == 3:
        path1 = sys.argv[1]
        path2 = sys.argv[2]
    else:
        print('usage:')
        print(' %s [path1] [path2]' % (sys.argv[0]))
        exit(1)

    print('compare inner result in %s %s' % (path1, path2))
    exit(compare(path1, path2))
