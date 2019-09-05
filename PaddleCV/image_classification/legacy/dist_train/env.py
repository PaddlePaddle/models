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


def dist_env():
    """
    Return a dict of all variable that distributed training may use.
    NOTE: you may rewrite this function to suit your cluster environments.
    """
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    num_trainers = 1
    training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
    assert (training_role == "PSERVER" or training_role == "TRAINER")

    # - PADDLE_TRAINER_ENDPOINTS means nccl2 mode.
    # - PADDLE_PSERVER_ENDPOINTS means pserver mode.
    # - PADDLE_CURRENT_ENDPOINT means current process endpoint.
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    pserver_endpoints = os.getenv("PADDLE_PSERVER_ENDPOINTS")
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
    if trainer_endpoints:
        trainer_endpoints = trainer_endpoints.split(",")
        num_trainers = len(trainer_endpoints)
    elif pserver_endpoints:
        num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))

    return {
        "trainer_id": trainer_id,
        "num_trainers": num_trainers,
        "current_endpoint": current_endpoint,
        "training_role": training_role,
        "pserver_endpoints": pserver_endpoints,
        "trainer_endpoints": trainer_endpoints
    }
