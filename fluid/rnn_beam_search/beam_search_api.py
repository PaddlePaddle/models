import contextlib
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.framework import Variable
from paddle.fluid import framework, unique_name
from paddle.fluid.layer_helper import LayerHelper
import paddle.fluid.core as core


class DecoderType:
    TRAINING = 1
    BEAM_SEARCH = 2


class InitState(object):
    def __init__(self,
                 init=None,
                 shape=None,
                 value=0.0,
                 need_reorder=False,
                 dtype='float32'):
        self._init = init
        self._shape = shape
        self._value = value
        self._need_reorder = need_reorder
        self._dtype = dtype

    @property
    def value(self):
        return self._init  # may create a LoDTensor

    @property
    def need_reorder(self):
        return self._need_reorder


class MemoryState(object):
    def __init__(self, state_name, rnn_obj, init_state):
        self._state_name = state_name  # each is a rnn.memory
        self._rnn_obj = rnn_obj
        self._state_mem = self._rnn_obj.memory(
            init=init_state.value, need_reorder=init_state.need_reorder)

    def get_state(self):
        return self._state_mem

    def update_state(self, state):
        self._rnn_obj.update_memory(self._state_mem, state)


class ArrayState(object):
    def __init__(self, state_name, block, init_state):
        self._state_name = state_name
        self._block = block

        self._state_array = self._block.create_var(
            name=unique_name.generate('array_state_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=init_state.value.dtype)

        self._counter = self._block.create_var(
            name=unique_name.generate('array_state_counter'),
            type=core.VarDesc.VarType.LOD_TENSOR,
            dtype='int64')

        # initialize counter
        self._block.append_op(
            type='fill_constant',
            inputs={},
            outputs={'Out': [self._counter]},
            attrs={
                'shape': [1],
                'dtype': self._counter.dtype,
                'value': float(0.0),
                'force_cpu': True
            })

        self._counter.stop_gradient = True

        # write initial state
        block.append_op(
            type='write_to_array',
            inputs={'X': init_state.value,
                    'I': self._counter},
            outputs={'Out': self._state_array})

    def get_state(self):
        state = layers.array_read(array=self._state_array, i=self._counter)
        return state

    def update_state(self, state):
        layers.increment(x=self._counter, value=1, in_place=True)
        layers.array_write(state, array=self._state_array, i=self._counter)


class StateCell(object):
    def __init__(self, cell_size, inputs, states, name=None):
        self._helper = LayerHelper('state_cell', name=name)
        self._cur_states = {}
        self._state_names = []
        self._states_holder = {}
        for state_name, state in states.items():
            if not isinstance(state, InitState):
                raise ValueError('state must be an InitState object.')
            self._cur_states[state_name] = state
            self._state_names.append(state_name)
        self._inputs = inputs  # inputs is place holder here
        self._cur_decoder_obj = None
        self._in_decoder = False
        self._states_holder = {}
        self._switched_decoder = False
        self._state_updater = None

    def enter_decoder(self, decoder_obj):
        if self._in_decoder == True or self._cur_decoder_obj is not None:
            raise ValueError('StateCell has already entered a decoder.')
        self._in_decoder = True
        self._cur_decoder_obj = decoder_obj
        self._switched_decoder = False

    def _switch_decoder(self):  # lazy switch
        if self._in_decoder == False:
            raise ValueError('StateCell must be enter a decoder.')

        if self._switched_decoder == True:
            raise ValueError('StateCell already done switching.')

        for state_name in self._state_names:
            if state_name not in self._states_holder:
                state = self._cur_states[state_name]

                if not isinstance(state, InitState):
                    raise ValueError('Current type of state is %s, should be '
                                     'an InitState object.' % type(state))

                self._states_holder[state_name] = {}

                if self._cur_decoder_obj.type == DecoderType.TRAINING:
                    self._states_holder[state_name][id(self._cur_decoder_obj)] \
                            = MemoryState(state_name,
                                          self._cur_decoder_obj.dynamic_rnn,
                                          state)
                elif self._cur_decoder_obj.type == DecoderType.BEAM_SEARCH:
                    self._states_holder[state_name][id(self._cur_decoder_obj)] \
                            = ArrayState(state_name,
                                         self._cur_decoder_obj.parent_block(),
                                         state)
                else:
                    raise ValueError('Unknown decoder type, only support '
                                     '[TRAINING, BEAM_SEARCH]')

            # Read back, since current state should be LoDTensor
            self._cur_states[state_name] = \
                    self._states_holder[state_name][id(self._cur_decoder_obj)].get_state()

        self._switched_decoder = True

    def get_state(self, state_name):
        if self._in_decoder and not self._switched_decoder:
            self._switch_decoder()

        if state_name not in self._cur_states:
            raise ValueError(
                'Unknown state %s. Please make sure _switch_decoder() '
                'invoked.' % state_name)

        return self._cur_states[state_name]

    def get_input(self, input_name):
        if input_name not in self._inputs or self._inputs[input_name] is None:
            raise ValueError('Invalid input %s.' % input_name)
        return self._inputs[input_name]

    def set_state(self, state_name, state_value):
        self._cur_states[state_name] = state_value

    def state_updater(self, updater):
        self._state_updater = updater

        def _decorator(state_cell):
            if state_cell == self:
                raise TypeError('Updater should only accept a StateCell object '
                                'as argument.')
            updater(state_cell)

        return _decorator

    def compute_state(self, inputs):
        if self._in_decoder and not self._switched_decoder:
            self._switch_decoder()

        for input_name, input_value in inputs.items():
            if input_name not in self._inputs:
                raise ValueError('Unknown input %s. '
                                 'Please make sure %s in input '
                                 'place holder.' % (input_name, input_name))
            self._inputs[input_name] = input_value
        self._state_updater(self)

    def update_states(self):
        if self._in_decoder and not self._switched_decoder:
            self._switched_decoder()

        for state_name, decoder_state in self._states_holder.items():
            if id(self._cur_decoder_obj) not in decoder_state:
                raise ValueError('Unknown decoder object, please make sure '
                                 'switch_decoder been invoked.')
            decoder_state[id(self._cur_decoder_obj)].update_state(
                self._cur_states[state_name])

    def leave_decoder(self, decoder_obj):
        if self._in_decoder == False:
            raise ValueError('StateCell not in decoder, '
                             'invlid leaving operation.')

        if self._cur_decoder_obj != decoder_obj:
            raise ValueError('Inconsist decoder object in StateCell.')

        self._in_decoder = False
        self._cur_decoder_obj = None
        self._switched_decoder = False


class TrainingDecoder(object):
    BEFORE_DECODER = 0
    IN_DECODER = 1
    AFTER_DECODER = 2

    def __init__(self, state_cell, name=None):
        self._helper = LayerHelper('training_decoder', name=name)
        self._status = TrainingDecoder.BEFORE_DECODER
        self._dynamic_rnn = layers.DynamicRNN()
        self._type = DecoderType.TRAINING
        self._state_cell = state_cell
        self._state_cell.enter_decoder(self)

    @contextlib.contextmanager
    def block(self):
        if self._status != TrainingDecoder.BEFORE_DECODER:
            raise ValueError('decoder.block() can only be invoked once')
        self._status = TrainingDecoder.IN_DECODER
        with self._dynamic_rnn.block():
            yield
        self._status = TrainingDecoder.AFTER_DECODER
        self._state_cell.leave_decoder(self)

    @property
    def state_cell(self):
        self._assert_in_decoder_block('state_cell')
        return self._state_cell

    @property
    def dynamic_rnn(self):
        return self._dynamic_rnn

    @property
    def type(self):
        return self._type

    def step_input(self, x):
        self._assert_in_decoder_block('step_input')
        return self._dynamic_rnn.step_input(x)

    def static_input(self, x):
        self._assert_in_decoder_block('static_input')
        return self._dynamic_rnn.static_input(x)

    def __call__(self, *args, **kwargs):
        if self._status != TrainingDecoder.AFTER_DECODER:
            raise ValueError('Output of training decoder can only be visited '
                             'outside the block.')
        return self._dynamic_rnn(*args, **kwargs)

    def output(self, *outputs):
        self._assert_in_decoder_block('output')
        self._dynamic_rnn.output(*outputs)

    def _assert_in_decoder_block(self, method):
        if self._status != TrainingDecoder.IN_DECODER:
            raise ValueError('%s should be invoked inside block of '
                             'TrainingDecoder object.' % method)


class BeamSearchDecoder(object):
    BEFORE_BEAM_SEARCH_DECODER = 0
    IN_BEAM_SEARCH_DECODER = 1
    AFTER_BEAM_SEARCH_DECODER = 2

    def __init__(self, state_cell, max_len, name=None):
        self._helper = LayerHelper('beam_search_decoder', name=name)
        self._counter = layers.zeros(shape=[1], dtype='int64')
        self._counter.stop_gradient = True
        self._type = DecoderType.BEAM_SEARCH
        self._max_len = layers.fill_constant(
            shape=[1], dtype='int64', value=max_len)
        self._cond = layers.less_than(
            x=self._counter,
            y=layers.fill_constant(
                shape=[1], dtype='int64', value=max_len))
        self._while_op = layers.While(self._cond)
        self._state_cell = state_cell
        self._state_cell.enter_decoder(self)
        self._status = BeamSearchDecoder.BEFORE_BEAM_SEARCH_DECODER
        self._zero_idx = layers.fill_constant(
            shape=[1], value=0, dtype='int64', force_cpu=True)
        self._array_dict = {}
        self._array_link = []
        self._ids_array = None
        self._scores_array = None

    @contextlib.contextmanager
    def block(self):
        if self._status != BeamSearchDecoder.BEFORE_BEAM_SEARCH_DECODER:
            raise ValueError('block() can only be invoke once.')

        self._status = BeamSearchDecoder.IN_BEAM_SEARCH_DECODER

        with self._while_op.block():
            yield

            layers.increment(x=self._counter, value=1.0, in_place=True)

            for value, array in self._array_link:
                layers.array_write(x=value, i=self._counter, array=array)

            layers.less_than(x=self._counter, y=self._max_len, cond=self._cond)

        self._status = BeamSearchDecoder.AFTER_BEAM_SEARCH_DECODER
        self._state_cell.leave_decoder(self)

    @property
    def type(self):
        return self._type

    # init must be provided
    def read_array(self, init, is_ids=False, is_scores=False):
        self._assert_in_decoder_block('read_array')

        if is_ids == True and is_scores == True:
            raise ValueError('Shouldn\'t mark current array be ids array and'
                             'scores array at the same time.')

        if not isinstance(init, Variable):
            raise TypeError('The input argument `init` must be a Variable.')

        parent_block = self.parent_block()
        array = parent_block.create_var(
            name=unique_name.generate('beam_search_decoder_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=init.dtype)
        parent_block.append_op(
            type='write_to_array',
            inputs={'X': init,
                    'I': self._zero_idx},
            outputs={'Out': array})

        if is_ids == True:
            self._ids_array = array
        elif is_scores == True:
            self._scores_array = array

        read_value = layers.array_read(array=array, i=self._counter)
        self._array_dict[read_value.name] = array
        return read_value

    def update_array(self, array, value):
        self._assert_in_decoder_block('update_array')

        if not isinstance(array, Variable):
            raise TypeError(
                'The input argument `array` of  must be a Variable.')
        if not isinstance(value, Variable):
            raise TypeError('The input argument `value` of must be a Variable.')

        array = self._array_dict.get(array.name, None)
        if array is None:
            raise ValueError('Please invoke read_array before update_array.')
        self._array_link.append((value, array))

    def __call__(self):
        if self._status != BeamSearchDecoder.AFTER_BEAM_SEARCH_DECODER:
            raise ValueError('Output of BeamSearchDecoder object can '
                             'only be visited outside the block.')
        return layers.beam_search_decode(
            ids=self._ids_array, scores=self._scores_array)

    @property
    def state_cell(self):
        self._assert_in_decoder_block('state_cell')
        return self._state_cell

    def parent_block(self):
        program = self._helper.main_program
        parent_block_idx = program.current_block().parent_idx
        if parent_block_idx < 0:
            raise ValueError('Invlid block with index %d.' % parent_block_idx)
        parent_block = program.block(parent_block_idx)
        return parent_block

    def _assert_in_decoder_block(self, method):
        if self._status != BeamSearchDecoder.IN_BEAM_SEARCH_DECODER:
            raise ValueError('%s should be invoked inside block of '
                             'BeamSearchDecoder object.' % method)
