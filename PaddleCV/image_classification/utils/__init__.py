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
from .learning_rate import cosine_decay, lr_warmup, cosine_decay_with_warmup, Decay
from .fp16_utils import create_master_params_grads, master_param_to_train_param
from .utility import add_arguments, print_arguments, parse_args, check_args, init_model, save_model, create_pyreader, print_info, best_strategy_compiled
from .metrics import Metrics, GoogLeNet_Metrics, Mixup_Metrics, create_metrics
