import os
import sys
import argparse
import importlib
import multiprocessing
import paddle
import cv2 as cv

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import ltr.admin.settings as ws_settings


def run_training(train_module, train_name):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
    """
    # set single threads in opencv
    cv.setNumThreads(0)

    print('Training:  {}  {}'.format(train_module, train_name))

    settings = ws_settings.Settings()

    if settings.env.workspace_dir == '':
        raise Exception('Setup your workspace_dir in "ltr/admin/local.py".')

    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'ltr/{}/{}'.format(train_module, train_name)

    expr_module = importlib.import_module('ltr.train_settings.{}.{}'.format(
        train_module, train_name))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(
        description='Run a train scripts in train_settings.')
    parser.add_argument(
        'train_module',
        type=str,
        help='Name of module in the "train_settings/" folder.')
    parser.add_argument(
        'train_name', type=str, help='Name of the train settings file.')

    args = parser.parse_args()

    run_training(args.train_module, args.train_name)


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    multiprocessing.set_start_method('spawn', force=True)
    main()
