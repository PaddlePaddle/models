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
"""Get latency lookup table."""
from __future__ import print_function

import re
import argparse
import subprocess

from light_nas_space import get_all_ops


def get_args():
    """Get arguments.

    Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--latency_lookup_table_path',
        default='latency_lookup_table.txt',
        help='Output latency lookup table path.')
    parser.add_argument(
        '--platform', default='android', help='Platform: android/ios/custom.')
    parser.add_argument('--threads', type=int, default=1, help='Threads.')
    parser.add_argument(
        '--test_iter',
        type=int,
        default=100,
        help='Running times of op when estimating latency.')
    args = parser.parse_args()
    return args


def get_op_latency(op, platform):
    """Get op latency.

    Args:
        op: list, a list of str represents the op and its parameters.
        platform: str, platform name.

    Returns:
        float, op latency.
    """
    if platform == 'android':
        commands = 'adb shell "cd /data/local/tmp/bin && LD_LIBRARY_PATH=. ./get_{}_latency \'{}\'"'.format(
            op[0], ' '.join(op[1:]))
        proc = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
        out = proc.communicate()[0]
        out = [_ for _ in out.decode().split('\n') if 'Latency' in _][-1]
        out = re.findall(r'\d+\.?\d*', out)[0]
        out = float(out)
    elif platform == 'ios':
        print('Please refer the usage doc to get iOS latency lookup table')
        out = 0
    else:
        print('Please define `get_op_latency` for {} platform'.format(platform))
        out = 0
    return out


def main():
    """main."""
    args = get_args()
    ops = get_all_ops()
    fid = open(args.latency_lookup_table_path, 'w')
    for op in ops:
        op = [str(item) for item in op]
        latency = get_op_latency(
            op[:1] + [str(args.threads), str(args.test_iter)] + op[1:],
            args.platform)
        fid.write('{} {}\n'.format(' '.join(op), latency))
    fid.close()


if __name__ == '__main__':
    main()
