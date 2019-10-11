#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import paddle.fluid as fluid


def nccl2_prepare(args, startup_program, main_program):
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "127.0.0.1:6170")
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "127.0.0.1:6170")

    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    if args.nccl_comm_num > 1:
        config.nccl_comm_num = args.nccl_comm_num
    trainers_num = len(trainer_endpoints.split(','))
    if args.use_hallreduce and trainers_num > 8:
        config.use_hierarchical_allreduce = args.use_hallreduce
        config.hierarchical_allreduce_inter_nranks = 8

        assert config.hierarchical_allreduce_inter_nranks > 1
        assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0
        config.hierarchical_allreduce_exter_nranks = \
            trainers_num / config.hierarchical_allreduce_inter_nranks

    t = fluid.DistributeTranspiler(config=config)

    t.transpile(
        trainer_id,
        trainers=trainer_endpoints,
        current_endpoint=current_endpoint,
        startup_program=startup_program,
        program=main_program)


def dist_env():
    """
    Return a dict of all variable that distributed training may use.
    NOTE: you may rewrite this function to suit your cluster environments.
    """
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "127.0.0.1:6170")
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "127.0.0.1:6170")
    trainer_endpoints = trainer_endpoints.split(',')

    assert num_trainers == len(trainer_endpoints), \
        "num_trainers must equal to len(trainer_endpoints)."
    
    return {
        "trainer_id": trainer_id,
        "num_trainers": num_trainers,
        "current_endpoint": current_endpoint,
        "trainer_endpoints": trainer_endpoints
    }
