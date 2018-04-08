"""
    External neural memory class.
"""
import paddle.v2 as paddle


class ExternalMemory(object):
    """External neural memory class.

    A simplified Neural Turing Machines (NTM) with only content-based
    addressing (including content addressing and interpolation, but excluding
    convolutional shift and sharpening). It serves as an external differential
    memory bank, with differential write/read head controllers to store
    and read information dynamically. Simple feedforward networks are
    used as the write/read head controllers.

    The ExternalMemory class could be utilized by many neural network structures
    to easily expand their memory bandwidth and accomplish a long-term memory
    handling. Besides, some existing mechanism can be realized directly with
    the ExternalMemory class, e.g. the attention mechanism in Seq2Seq (i.e. an
    unbounded external memory).

    Besides, the ExternalMemory class must be used together with
    paddle.layer.recurrent_group (within its step function). It can never be
    used in a standalone manner.

    For more details, please refer to
    `Neural Turing Machines <https://arxiv.org/abs/1410.5401>`_.

    :param name: Memory name.
    :type name: basestring
    :param mem_slot_size: Size of memory slot/vector.
    :type mem_slot_size: int
    :param boot_layer: Boot layer for initializing the external memory. The
                       sequence layer has sequence length indicating the number
                       of memory slots, and size as memory slot size.
    :type boot_layer: LayerOutput
    :param initial_weight: Initializer for addressing weights.
    :type initial_weight: LayerOutput
    :param readonly: If true, the memory is read-only, and write function cannot
                     be called. Default is false.
    :type readonly: bool
    :param enable_interpolation: If set true, the read/write addressing weights
                                 will be interpolated with the weights in the
                                 last step, with the affine coefficients being
                                 a learnable gate function.
    :type enable_interpolation: bool
    """

    def __init__(self,
                 name,
                 mem_slot_size,
                 boot_layer,
                 initial_weight,
                 readonly=False,
                 enable_interpolation=True):
        self.name = name
        self.mem_slot_size = mem_slot_size
        self.readonly = readonly
        self.enable_interpolation = enable_interpolation
        self.external_memory = paddle.layer.memory(
            name=self.name, size=self.mem_slot_size, boot_layer=boot_layer)
        self.initial_weight = initial_weight
        # set memory to constant when readonly=True
        if self.readonly:
            self.updated_external_memory = paddle.layer.mixed(
                name=self.name,
                input=[
                    paddle.layer.identity_projection(input=self.external_memory)
                ],
                size=self.mem_slot_size)

    def _content_addressing(self, key_vector):
        """Get write/read head's addressing weights via content-based addressing.
        """
        # content-based addressing: a=tanh(W*M + U*key)
        key_projection = paddle.layer.fc(input=key_vector,
                                         size=self.mem_slot_size,
                                         act=paddle.activation.Linear(),
                                         bias_attr=False)
        key_proj_expanded = paddle.layer.expand(
            input=key_projection, expand_as=self.external_memory)
        memory_projection = paddle.layer.fc(input=self.external_memory,
                                            size=self.mem_slot_size,
                                            act=paddle.activation.Linear(),
                                            bias_attr=False)
        merged_projection = paddle.layer.addto(
            input=[key_proj_expanded, memory_projection],
            act=paddle.activation.Tanh())
        # softmax addressing weight: w=softmax(v^T a)
        addressing_weight = paddle.layer.fc(
            input=merged_projection,
            size=1,
            act=paddle.activation.SequenceSoftmax(),
            bias_attr=False)
        return addressing_weight

    def _interpolation(self, head_name, key_vector, addressing_weight):
        """Interpolate between previous and current addressing weights.
        """
        # prepare interpolation scalar gate: g=sigmoid(W*key)
        gate = paddle.layer.fc(input=key_vector,
                               size=1,
                               act=paddle.activation.Sigmoid(),
                               bias_attr=False)
        # interpolation: w_t = g*w_t+(1-g)*w_{t-1}
        last_addressing_weight = paddle.layer.memory(
            name=self.name + "_addressing_weight_" + head_name,
            size=1,
            boot_layer=self.initial_weight)
        interpolated_weight = paddle.layer.interpolation(
            name=self.name + "_addressing_weight_" + head_name,
            input=[last_addressing_weight, addressing_weight],
            weight=paddle.layer.expand(
                input=gate, expand_as=addressing_weight))
        return interpolated_weight

    def _get_addressing_weight(self, head_name, key_vector):
        """Get final addressing weights for read/write heads, including content
        addressing and interpolation.
        """
        # current content-based addressing
        addressing_weight = self._content_addressing(key_vector)
        # interpolation with previous addresing weight
        if self.enable_interpolation:
            return self._interpolation(head_name, key_vector, addressing_weight)
        else:
            return addressing_weight

    def write(self, write_key):
        """Write onto the external memory.
        It cannot be called if "readonly" set True.

        :param write_key: Key vector for write heads to generate writing
                          content and addressing signals.
        :type write_key: LayerOutput
        """
        # check readonly
        if self.readonly:
            raise ValueError("ExternalMemory with readonly=True cannot write.")
        # get addressing weight for write head
        write_weight = self._get_addressing_weight("write_head", write_key)
        # prepare add_vector and erase_vector
        erase_vector = paddle.layer.fc(input=write_key,
                                       size=self.mem_slot_size,
                                       act=paddle.activation.Sigmoid(),
                                       bias_attr=False)
        add_vector = paddle.layer.fc(input=write_key,
                                     size=self.mem_slot_size,
                                     act=paddle.activation.Sigmoid(),
                                     bias_attr=False)
        erase_vector_expand = paddle.layer.expand(
            input=erase_vector, expand_as=self.external_memory)
        add_vector_expand = paddle.layer.expand(
            input=add_vector, expand_as=self.external_memory)
        # prepare scaled add part and erase part
        scaled_erase_vector_expand = paddle.layer.scaling(
            weight=write_weight, input=erase_vector_expand)
        erase_memory_part = paddle.layer.mixed(
            input=paddle.layer.dotmul_operator(
                a=self.external_memory,
                b=scaled_erase_vector_expand,
                scale=-1.0))
        add_memory_part = paddle.layer.scaling(
            weight=write_weight, input=add_vector_expand)
        # update external memory
        self.updated_external_memory = paddle.layer.addto(
            input=[self.external_memory, add_memory_part, erase_memory_part],
            name=self.name)

    def read(self, read_key):
        """Read from the external memory.

        :param write_key: Key vector for read head to generate addressing
                          signals.
        :type write_key: LayerOutput
        :return: Content (vector) read from external memory.
        :rtype: LayerOutput
        """
        # get addressing weight for write head
        read_weight = self._get_addressing_weight("read_head", read_key)
        # read content from external memory
        scaled = paddle.layer.scaling(
            weight=read_weight, input=self.updated_external_memory)
        return paddle.layer.pooling(
            input=scaled, pooling_type=paddle.pooling.Sum())
