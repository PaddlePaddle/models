import argparse
import distutils.util
import numpy as np
import re


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def log_info(s):
    try:
        log_file, log_label = s.split(':')
    except:
        raise argparse.ArgumentTypeError('Format of log info '
                                         'should be "log_file:log_label".')


parser = argparse.ArgumentParser("Tools to plot learning curves.")
parser.add_argument(
    '--log_infos',
    type=str,
    nargs='+',
    required=True,
    help='Log infos to indicate whose curves to plot. '
    'Format is "log_file:log_label".')
parser.add_argument(
    '--plot_item',
    type=str,
    choices=['sum loss', 'avg loss', 'ppl'],
    default='ppl',
    help='Item to plot. (default: %(default)d)')
parser.add_argument(
    '--save_path',
    type=str,
    default='./curve_comparison.png',
    help='Path to save plotting image. (default: %(default)d)')
parser.add_argument(
    '--plot_validation',
    type=distutils.util.strtobool,
    default=True,
    help='Whether plot for validation data. (default: %(default)d)')
parser.add_argument(
    '--whether_show',
    type=distutils.util.strtobool,
    default=True,
    help='Whether show the image. (default: %(default)d)')
args = parser.parse_args()

import matplotlib
if args.whether_show == False:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def item_to_name(item):
    if item == 'sum loss':
        return 'SUM Loss'
    elif item == 'avg loss':
        return 'AVG Loss'
    elif item == 'ppl':
        return 'PPL'


def parse_log(log_file, item):
    iter_num = 0
    train_item = []
    val_item = []
    iter_num = 0
    train_pattern = r'.*, %s: ([^,]+),.*' % item
    val_pattern = r'.*, %s: ([^,]+),.*' % ('val ' + item)
    for line in open(log_file):
        line = line.strip()
        line += ','
        train_matched = re.match(train_pattern, line)
        val_matched = re.match(val_pattern, line)
        if train_matched is not None:
            train_item.append([iter_num, float(train_matched.groups()[0])])
            iter_num += 1
        elif val_matched is not None:
            val_item.append([iter_num, float(val_matched.groups()[0])])

    return np.array(train_item), np.array(val_item)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

whether_plot_val = False

for log_info in args.log_infos:
    log_file, log_label = log_info.split(':')
    train_item, val_item = parse_log(log_file, args.plot_item)

    if args.plot_validation and len(val_item) > 0:
        whether_plot_val = True

    plt.plot(
        train_item[:, 0],
        train_item[:, 1],
        label=log_label + '-' + item_to_name(args.plot_item),
        linewidth=0.5)

    if whether_plot_val:
        plt.plot(
            val_item[:, 0],
            val_item[:, 1],
            label=log_label + '-' + 'Val ' + item_to_name(args.plot_item),
            linewidth=1.0,
            linestyle='-.')

colormap = plt.cm.gist_ncar  #nipy_spectral, Set1,Paired

colors = np.linspace(0.1, 0.8, len(ax.lines))
colors = [colormap(i) for i in colors]

for i, j in enumerate(ax.lines):
    j.set_color(colors[i])

ax.legend(loc='best')
ax.set_xlabel('Iteration Number', fontsize=10)
ax.set_ylabel(item_to_name(args.plot_item), fontsize=10)
plt.savefig(args.save_path, bbox_inches='tight')
if args.whether_show:
    plt.show()
