"""Parse the log for tuning and plot error surface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import argparse
import functools
import _init_paths
from utils.utility import add_arguments, print_arguments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("log_path", str, '', "log path for parsing")
add_arg("fig_name", str, 'error_surface.png', "name of output figure")
args = parser.parse_args()


def plot_error_surface(num_alphas, alphas, betas, error_rate_type, err_ave):
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)

    num_betas = len(alphas) // num_alphas
    alphas_2d = np.reshape(alphas, (num_alphas, num_betas))
    betas_2d = np.reshape(betas, (num_alphas, num_betas))
    err_ave_2d = np.reshape(err_ave, (num_alphas, num_betas))

    ax.plot_surface(
        alphas_2d,
        betas_2d,
        err_ave_2d,
        rstride=1,
        cstride=1,
        alpha=0.8,
        cmap='rainbow')
    z_label = 'WER' if error_rate_type == 'wer' else 'CER'
    ax.set_xlabel('alpha', fontsize=12)
    ax.set_ylabel('beta', fontsize=12)
    ax.set_zlabel(z_label, fontsize=12)
    plt.savefig(args.fig_name)
    plt.show()


def parse_log():
    if not os.path.isfile(args.log_path):
        raise IOError("Invaid model path: %s" % args.log_path)

    error_rate_type = None
    num_alphas, num_betas = 0, 0
    alphas, betas, err_ave = [], [], []

    err_rate_pat = re.compile(
        '\(alpha, beta\) = '
        '\([-+]?\d+(?:\.\d+)?, [-+]?\d+(?:\.\d+)?\), \[[wcer]')
    num_pat = re.compile(r'[-+]?\d+(?:\.\d+)?')

    with open(args.log_path, "r") as log_file:
        line = log_file.readline()
        while line:
            if err_rate_pat.match(line) is not None:
                triple = num_pat.findall(line)
                alphas.append(float(triple[0]))
                betas.append(float(triple[1]))
                err_ave.append(float(triple[2]))
            elif line.find("error_rate_type:") != -1:
                error_rate_type = line.strip().split()[1]
            elif line.find("num_alphas:") != -1:
                num_alphas = int(line.strip().split()[1])
            elif line.find("num_betas:") != -1:
                num_betas = int(line.strip().split()[1])
            line = log_file.readline()

    if error_rate_type == None:
        raise ValueError("Illegal log format, cannot find error_rate_type")

    if num_alphas <= 0:
        raise ValueError("Illegal log format, invalid num_alphas")

    if num_betas <= 0:
        raise ValueError("Illegal log format, invalid num_betas")

    if alphas == []:
        raise ValueError("Illegal log format, cannot find grid search result")

    if num_alphas * num_betas != len(alphas):
        raise ValueError("Illegal log format, data's shape mismatches")

    return num_alphas, alphas, betas, error_rate_type, err_ave,


def main():
    print_arguments(args)
    num_alphas, alphas, betas, error_rate_type, err_ave = parse_log()
    plot_error_surface(num_alphas, alphas, betas, error_rate_type, err_ave)


if __name__ == '__main__':
    main()
