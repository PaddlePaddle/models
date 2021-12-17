# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle_serving_server.web_service import WebService, Op


class TIPCExampleOp(Op):
    """TIPCExampleOp
    
    ExampleOp for serving server. You can rename by yourself. 
    """

    def init_op(self):
        """init_op

        Initialize the class.

        Args: None

        Returns: None
        """
        pass

    def preprocess(self, input_dicts, data_id, log_id):
        """preprocess
        
        In preprocess stage, assembling data for process stage. users can 
        override this function for model feed features.

        Args:
            input_dicts: input data to be preprocessed
            data_id: inner unique id, increase auto
            log_id: global unique id for RTT, 0 default

        Return:
            output_data: data for process stage
            is_skip_process: skip process stage or not, False default
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception. 
            prod_errinfo: "" default
        """
        pass

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        """postprocess

        In postprocess stage, assemble data for next op or output.
        Args:
            input_data: data returned in preprocess stage, dict(for single predict) or list(for batch predict)
            fetch_data: data returned in process stage, dict(for single predict) or list(for batch predict)
            data_id: inner unique id, increase auto
            log_id: logid, 0 default

        Returns: 
            fetch_dict: fetch result must be dict type.
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception.
            prod_errinfo: "" default
        """
        # postprocess for the service output
        pass


class TIPCExampleService(WebService):
    """TIPCExampleService
    
    Service class to define the Serving OP.
    """

    def get_pipeline_response(self, read_op):
        tipc_example_op = TIPCExampleOp(
            name="tipc_example", input_ops=[read_op])
        return tipc_example_op


# define the service class
uci_service = TIPCExampleService(name="tipc_example")
# load config and prepare the service
uci_service.prepare_pipeline_config("config.yml")
# start the service
uci_service.run_service()
