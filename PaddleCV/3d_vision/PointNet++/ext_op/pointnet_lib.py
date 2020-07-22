#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

file_dir = os.path.dirname(os.path.abspath(__file__))
fluid.load_op_library(os.path.join(file_dir, 'src/pointnet_lib.so'))

from paddle.fluid.layer_helper import LayerHelper

__all__ = ['three_nn', 'three_interp', 'query_ball', 'gather_point',
            'farthest_point_sampling', 'group_points']


def three_nn(input, known, eps=1e-10, name=None):
    """
    **Three Nearest Neighbor Layer**

    This operator samples the top-3 nearest neighbor of each point
    coordinates specified by Input(X) between known point coordinates
    specified by Input(Known) and calcualte the distance between these
    nearest neighbors.

    Args:
        input (Variable): The input tensor of three_nn operator. This
                          is a 3-D tensor with shape of [B, N, 3].
        known (Variable): The input tensor of known points of three_nn
                          operator. This is a 3-D tensor with shape of
                          [B, M, 3].
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        distance (Variable): The output distance tensor of three_nn operator.
                             This is a 3-D tensor with shape of [B, N, 3].
        idx (Variable): The output index tensor of three_nn operator.
                             This is a 3-D tensor with shape of [B, N, 3].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 16, 3], dtype='float32')
            known = fluid.data(name='known', shape=[None, 32, 3], dtype='float32')
            distance, idx = fluid.layers.three_nn(input, known)
    """
    helper = LayerHelper('three_nn', **locals())
    dtype = helper.input_dtype()
    dist = helper.create_variable_for_type_inference(dtype)
    idx = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="three_nn",
        inputs={"X": input,
                "Known": known},
        outputs={"Distance": dist,
                 "Idx": idx},
        attrs={'eps': eps})
    return (dist, idx)


def three_interp(input, weight, idx, name=None):
    """
    **Three Interpolate Layer**

    This operator calculate interpolate results from input, weight and
    index.

    Args:
        input (Variable): The input tensor of three_interp operator. This
                          is a 3-D tensor with shape of [B, M, C].
        weight (Variable): The weight tensor of three_interp operator. This
                          is a 3-D tensor with shape of [B, N, 3].
        idx (Variable): The index tensor of three_interp operator. This
                          is a 3-D tensor with shape of [B, N, 3].
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output (Variable): The output tensor of three_interp operator.
                             This is a 3-D tensor with shape of [B, N, C].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 16, 3], dtype='float32')
            weight = fluid.data(name='weight', shape=[None, 32, 3], dtype='float32')
            index = fluid.data(name='index', shape=[None, 32, 3], dtype='int32')
            out = fluid.layers.three_interp(x, weight, index)
    """
    helper = LayerHelper('three_interp', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="three_interp",
        inputs={"X": input,
                "Weight": weight,
                "Idx": idx},
        outputs={"Out": out, })
    return out


def query_ball(input, new_points, radius, n_sample):
    """
    **Query Ball Layer**

    Output is a tensor with the indicies of the features that form the query balls.

    Args:
        input(Variable): XYZ coordinates of features with shape of [B,N,3].
        new_points(Variable): Centers coordinates of the ball query with shape of [B,M,3].
        radius(float|Variable): Radius of the balls.
        n_sample(int|Variable): Maximum number of features in the balls.
    Return:
        output(Variable): Tensor with the indicies of the features that form the query balls,with shape of [B,M,n_sample]

    Examples:
        .. code-block::python

            import paddle.fluid as fluid
            x = fluid.data(name='points',shape=[None,5,3],dtype='float32')
            new_points = fluid.data(name='new_points', shape=[None,2,3], dtype='float32')
            output = fluid.layers.query_ball(x,new_points,radius=4.0,n_sample=5)



    """
    helper = LayerHelper('query_ball', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="query_ball",
        inputs={"Points": input,
                "New_Points": new_points},
        attrs={"N_sample": n_sample,
               "Radius": radius},
        outputs={"Output": out})
    return out


def farthest_point_sampling(input, sampled_point_num):
    '''
    Sampling point based on its max eucliden distance with other points. 
    
    Args:
        input (Variable): input point cloud dataset with shape (B, N, 3)
            B is batch size, N is points's nums, 3 is (x,y,z) coordinate
        sampled_point_num (int): sampled points's nums

    Retrun:
        output (Variable): return sampled points with shape (B, M)
            B is batch size, M is points's nums

    Examples:
        .. code-block:: python
        x = fluid.data(name='data', shape=(None ,100, 3), dtype='float32')
        sampled_points = fluid.layers.farthest_point_sampling(
            x, 50
        )
    '''

    helper = LayerHelper('farthest_point_sampling', **locals())
    dtype = input.dtype
    op_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='farthest_point_sampling',
        inputs={'X': input},
        outputs={'Output': op_out},
        attrs={'sampled_point_num': sampled_point_num})
    return op_out


def gather_point(input, index):
    """
    **Gather Point Layer**
    Output is obtained by gathering entries of X indexed by `index` 
    and concatenate them together.
    .. math::
        Out = X[Index]
    .. code-block:: text
        Given:
        X = [[1, 2, 3],
             [3, 4, 5],
             [5, 6, 7]]
        Index = [[1, 2]
        Then:
        Out = [[3, 4, 5],
               [5, 6, 7]]
    Args:
        input (Variable): The source input with rank>=1, This
                          is a 3-D tensor with shape of [B, N, 3].
        index (Variable): The index input with shape of [B, M].
      
    Returns:
        output (Variable): The output is a tensor with shape of [B,M].
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 5, 3], dtype='float32')
            index = fluid.data(name='index', shape=[None, 1], dtype='int32')
            output = fluid.layers.gather_point(x, index)
    """

    helper = LayerHelper('gather_point', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather_point",
        inputs={"X": input,
                "Index": index},
        outputs={"Output": out})
    return out


def group_points(input, idx, name=None):
    """
    **Group Points Layer**

    This operator group input points with index.

    Args:
        input (Variable): The input tensor of three_interp operator. This
                          is a 3-D tensor with shape of [B, N, C].
        idx (Variable): The index tensor of three_interp operator. This
                          is a 3-D tensor with shape of [B, M, S].
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output (Variable): The output tensor of three_interp operator.
                             This is a 4-D tensor with shape of [B, M, S, C].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 16, 3], dtype='float32')
            index = fluid.data(name='index', shape=[None, 32, 3], dtype='int32')
            out  = fluid.layers.group_points(x, index)
    """
    helper = LayerHelper('group_points', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="group_points",
        inputs={"X": input,
                "Idx": idx},
        outputs={"Out": out, })
    return out
