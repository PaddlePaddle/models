# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Get op latency."""
from __future__ import print_function

import re
import argparse
import subprocess


def get_args():
    """Get arguments.

    Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ops_path', default='ops.txt', help='Input ops path.')
    parser.add_argument(
        '--platform',
        default='paddlemobile-android',
        help='Platform: android/ios/custom.')
    parser.add_argument(
        '--ops_latency_path',
        default='ops_latency.txt',
        help='Output ops latency path.')
    args = parser.parse_args()
    return args


def get_op_latency(op, platform):
    """Get model latency.

    Args:
        op: list, a list of str represents the op and its parameters.
        platform: str, platform name.

    Returns:
        float, op latency.
    """
    if platform == 'anakin-android':
        commands = 'adb shell data/local/tmp/get_{}_latency {}'.format(
            op[0], ' '.join(op[1:]))
        proc = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
        out = proc.communicate()[1]
        out = [_ for _ in out.split('\n') if 'time' in _][-1]
        out = re.findall(r'\d+\.?\d*', out)[-2]
    elif platform == 'paddlemobile-android':
        commands = 'adb shell "cd /data/local/tmp/bin && export LD_LIBRARY_PATH=. && ./get_{}_latency \'{}\'"'.format(
            op[0], ' '.join(op[1:]))
        proc = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
        out = proc.communicate()[0]
        out = float(out)
    elif platform == 'ios':
        out = 0
    else:
        print('Please define `get_op_latency` for {} platform'.format(platform))
        out = 0
    return out


def main():
    """main."""
    args = get_args()
    ops = [line.split() for line in open(args.ops_path)]
    fid = open(args.ops_latency_path, 'w')
    for op in ops:
        latency = get_op_latency(op, args.platform)
        fid.write('{} {}\n'.format(' '.join(op), latency))
    fid.close()


if __name__ == '__main__':
    main()
