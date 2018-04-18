#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse

from kaffe import KaffeError, print_stderr
from kaffe.paddle import Transformer


def fatal_error(msg):
    """ fatal error encounted
    """
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    """ validate args
    """
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')


def convert(def_path, caffemodel_path, data_output_path, code_output_path,
            phase):
    """ convert caffe model to tf/paddle models
    """
    try:
        transformer = Transformer(def_path, caffemodel_path, phase=phase)
        print_stderr('Converting data...')
        if caffemodel_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path:
            print_stderr('Saving source...')
            with open(code_output_path, 'wb') as src_out:
                src_out.write(transformer.transform_source())
        print_stderr('set env variable before using converted model '\
                'if used custom_layers:')
        custom_pk_path = os.path.dirname(os.path.abspath(__file__))
        custom_pk_path = os.path.join(custom_pk_path, 'kaffe')
        print_stderr('export CAFFE2FLUID_CUSTOM_LAYERS=%s' % (custom_pk_path))
        print_stderr('Done.')
        return 0
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))

    return 1


def main():
    """ main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument(
        '--code-output-path', help='Save generated source to this path')
    parser.add_argument(
        '-p',
        '--phase',
        default='test',
        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    validate_arguments(args)
    return convert(args.def_path, args.caffemodel, args.data_output_path,
                   args.code_output_path, args.phase)


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
