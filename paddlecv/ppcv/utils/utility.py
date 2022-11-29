# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import importlib.util
import sys
import subprocess


def check_install(module_name, install_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f'Warnning! The {module_name} module is NOT installed')
        print(
            f'Try install {module_name} module automatically. You can also try to install manually by pip install {install_name}.'
        )
        python = sys.executable
        try:
            subprocess.check_call(
                [python, '-m', 'pip', 'install', install_name],
                stdout=subprocess.DEVNULL)
            print(f'The {module_name} module is now installed')
        except subprocess.CalledProcessError as exc:
            raise Exception(
                f"Install {module_name} failed, please install manually")
    else:
        print(f"{module_name} has been installed.")
