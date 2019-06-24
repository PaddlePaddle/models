#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.core import Compressor
from light_nas_space import LightNASSpace


def search():
    if not fluid.core.is_compiled_with_cuda():
        return

    space = LightNASSpace()

    startup_prog, train_prog, test_prog, train_metrics, test_metrics, train_reader, test_reader = space.create_net(
    )
    train_cost, train_acc1, train_acc5, global_lr = train_metrics
    test_cost, test_acc1, test_acc5 = test_metrics

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    val_fetch_list = [('acc_top1', test_acc1.name), ('acc_top5',
                                                     test_acc5.name)]
    train_fetch_list = [('loss', train_cost.name)]

    com_pass = Compressor(
        place,
        fluid.global_scope(),
        train_prog,
        train_reader=train_reader,
        train_feed_list=None,
        train_fetch_list=train_fetch_list,
        eval_program=test_prog,
        eval_reader=test_reader,
        eval_feed_list=None,
        eval_fetch_list=val_fetch_list,
        train_optimizer=None,
        search_space=space)
    com_pass.config('./compress.yaml')
    eval_graph = com_pass.run()


if __name__ == '__main__':
    search()
