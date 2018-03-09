import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.layer_helper import LayerHelper, unique_name
import paddle.v2.fluid.core as core


class DecoderType:
    TRAINING = 1
    BEAM_SEARCH = 2


class BasicRNNCell(object):
    def __init__(self, cell_size):
        self._size = cell_size

    def inject_decoder(self, decoder_obj, **kwargs):
        self._decoder_obj = decoder_obj

        if isinstance(decoder_obj, TrainingDecoder):
            self._decoder_type = DecoderType.TRAINING
            assert 'step_inputs' in kwargs
            self._step_inputs = kwargs['step_inputs']
            assert 'init_states' in kwargs
            self._init_states = kwargs['init_states']
        elif isinstance(decoder_obj, BeamSearchDecoder):
            self._decoder_type = DecoderType.BEAM_SEARCH
            assert 'init_states' in kwargs
            self._init_states = kwargs['init_states']

    def init_inputs(self):
        if self._decoder_type == DecoderType.TRAINING:
            assert len(self._step_inputs) == 1
            self._current_word = self._decoder_obj.rnn.step_input(
                self._step_inputs[0])

    def init_states(self):
        if self._decoder_type == DecoderType.TRAINING:
            assert len(self._init_states) == 1
            self._hidden_mem = self._decoder_obj.rnn.memory(
                init=self._init_states[0])
        elif self._decoder_type == DecoderType.BEAM_SEARCH:
            parent_block = self._decoder_obj.parent_block()
            self._hidden_mem = parent_block.create_var(
                name=unique_name('beam_search_basic_rnn_state'),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=self._init_states[0].dtype)
            parent_block.append_op(
                type='write_to_array',
                inputs={
                    'X': self._init_states[0],
                    'I': self._decoder_obj.counter
                },
                outputs={'Out': self._hidden_mem})

    def compute_current_state(self, **kwargs):
        if self._decoder_type == DecoderType.TRAINING:
            self._current_state = layers.fc(
                input=[self._current_word, self._hidden_mem], size=self._size)
        elif self._decoder_type == DecoderType.BEAM_SEARCH:
            prev_state = layers.array_read(
                array=self._hidden_mem, i=self._decoder_obj.counter)
            assert 'prev_scores' in kwargs
            prev_state_expanded = layers.sequence_expand(prev_state,
                                                         kwargs['prev_scores'])
            assert 'prev_ids_embedding' in kwargs
            self._current_state = layers.fc(
                input=[kwargs['prev_ids_embedding'], prev_state_expanded],
                size=self._size,
                act='tanh')

    def update_states(self):
        if self._decoder_type == DecoderType.TRAINING:
            self._decoder_obj.rnn.update_memory(self._hidden_mem,
                                                self._current_state)
        elif self._decoder_type == DecoderType.BEAM_SEARCH:
            layers.array_write(
                self._current_state,
                array=self._hidden_mem,
                i=self._decoder_obj.counter)

    def update_outputs(self):
        # may provide a output call back
        if self._decoder_type == DecoderType.TRAINING:
            self.calc_scores()
            self._decoder_obj.rnn.output(self._current_scores)

    def calc_scores(self):
        if self._decoder_type == DecoderType.TRAINING:
            self._current_scores = layers.fc(input=self._current_state,
                                             size=self._decoder_obj.label_dim,
                                             act='softmax')
            return self._current_scores
        elif self._decoder_type == DecoderType.BEAM_SEARCH:
            self._current_scores = layers.fc(input=self._current_state,
                                             size=self._decoder_obj.label_dim,
                                             act='softmax')
            return self._current_scores


class TrainingDecoder(object):
    def __init__(self,
                 cell_obj,
                 step_inputs,
                 label_dim,
                 static_inputs=None,
                 init_states=None):
        self.label_dim = label_dim
        self._helper = LayerHelper('training_decoder', name=name)

        if not isinstance(step_inputs, list):
            step_inputs = [step_inputs]

        if static_inputs is not None and not isinstance(static_inputs, list):
            static_inputs = [static_inputs]

        if init_states is not None and not isinstance(init_states, list):
            init_states = [init_states]

        self.rnn = layers.DynamicRNN()
        cell_obj.inject_decoder(
            self, step_inputs=step_inputs, init_states=init_states)

        with self.rnn.block():
            cell_obj.init_inputs()
            cell_obj.init_states()
            cell_obj.compute_current_state()
            cell_obj.update_states()
            cell_obj.update_outputs()

    def __call__(self):
        return self.rnn()


class BeamSearchDecoder(object):
    def __init__(self,
                 cell_obj,
                 init_ids,
                 init_scores,
                 init_states,
                 max_length,
                 label_dim,
                 eos_token,
                 beam_width,
                 embedding_layer,
                 name=None):
        self._helper = LayerHelper('beam_search_decoder', name=name)
        self.label_dim = label_dim

        if not isinstance(init_states, list):
            init_states = [init_states]

        array_len = layers.fill_constant(
            shape=[1], dtype='int64', value=max_length)
        self.counter = layers.zeros(shape=[1], dtype='int64')

        ids_array = layers.create_array('int64')
        layers.array_write(init_ids, array=ids_array, i=self.counter)

        scores_array = layers.create_array('float32')
        layers.array_write(init_scores, array=scores_array, i=self.counter)

        cond = layers.less_than(x=self.counter, y=array_len)

        cell_obj.inject_decoder(self, init_states=init_states)

        while_op = layers.While(cond=cond)
        with while_op.block():
            cell_obj.init_states()
            prev_ids = layers.array_read(array=ids_array, i=self.counter)
            prev_scores = layers.array_read(array=scores_array, i=self.counter)
            prev_ids_embedding = embedding_layer(prev_ids)

            cell_obj.compute_current_state(
                prev_scores=prev_scores, prev_ids_embedding=prev_ids_embedding)

            current_scores = cell_obj.calc_scores()

            topk_scores, topk_indices = layers.topk(
                current_scores, k=beam_width)
            selected_ids, selected_scores = layers.beam_search(
                prev_ids,
                topk_indices,
                topk_scores,
                beam_width,
                end_id=eos_token,
                level=0)

            layers.increment(x=self.counter, value=1, in_place=True)

            cell_obj.update_states()
            layers.array_write(selected_ids, array=ids_array, i=self.counter)
            layers.array_write(
                selected_scores, array=scores_array, i=self.counter)

            layers.less_than(x=self.counter, y=array_len, cond=cond)

        self._translation_ids, self._translation_scores = layers.beam_search_decode(
            ids=ids_array, scores=scores_array)

    def parent_block(self):
        program = self._helper.main_program
        parent_block_idx = program.current_block().parent_idx
        assert parent_block_idx >= 0
        parent_block = program.block(parent_block_idx)
        return parent_block

    def __call__(self):
        return self._translation_ids, self._translation_scores
