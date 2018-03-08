import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import contextlib
from paddle.v2.fluid.layer_helper import LayerHelper, unique_name
import paddle.v2.fluid.core as core


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


class MemoryState(object):
    def __init__(self, state_name, rnn_obj, init_state):
        self._state_name = state_name  # each is a rnn.memory
        self._rnn_obj = rnn_obj
        self._state_mem = self._rnn_obj.memory(init=init_state.value)

    def get_state(self):
        return self._state_mem

    def update_state(self, state):
        self._rnn_obj.update_memory(self._state_mem, state)


class ArrayState(object):
    def __init__(self, state_name, init_state):
        self._state_name = state_name
        self._counter = layers.zeros(shape=[1], dtype='int64')
        self._state_array = layers.create_array('int64')
        # write initial state
        layers.array_write(
            init_state.value,
            array=self._state_array,
            i=self._decoder_obj.counter)

    def get_state(self):
        state = layers.array_read(array=self._state_array, i=self._counter)
        return state

    def update_state(self, state):
        layers.increment(x=self._counter, value=1, in_place=True)
        layers.array_write(state, array=self._state_array, i=self._counter)


class StateCell(object):
    def __init__(self, cell_size, inputs, states, name=None):
        self._helper = LayerHelper("state_cell", name=name)
        self._cur_states = {}
        self._state_names = []
        for state_name, state in states.items():
            if not isinstance(state, InitState):
                raise ValueError("State must be an InitState object.")
            self._cur_states[state_name] = state
            self._state_names.append(state_name)
        self._inputs = inputs  # inputs is place holder here
        self._states_holder = {}
        self._cur_decoder_obj = None

    def switch_decoder(self, decoder_obj):
        self._cur_decoder_obj = decoder_obj
        for state_name in self._state_names:
            if state_name not in self._states_holder:
                state = self._cur_states[state_name]
                if not isinstance(state, InitState):
                    raise ValueError("Current type of state is %s, should be "
                                     "an InitState object." % type(state))
                if decoder_obj.type == DecoderType.TRAINING:
                    self._states_holder[state_name][decoder_obj] = \
                            MemoryState(state_name,
                                        decoder_obj.dynamic_rnn,
                                        state)
                elif decoder_obj.type == DecoderType.BEAM_SEARCH:
                    self._states_holder[state_name][decoder_obj] = \
                            ArrayState(state_name, state)
                else:
                    raise ValueError("Unknown decoder type, only support "
                                     "[TRAINING, BEAM_SEARCH]")
            # Read back, since current state should be LoDTensor
            self._cur_states[state_name] = \
                    self._states_holder[state_name][decoder_obj].get_state()

    def get_state(self, state_name):
        if state_name not in self._cur_states:
            raise ValueError(
                'Unknown state %s. Please make sure switch_decoder '
                'invoked.' % state_name)
        return self._cur_states[state_name]

    def get_input(self, input_name):
        if input_name not in self._inputs or self._inputs[input_name] is None:
            raise ValueError("Invalid input %s." % input_name)

    def set_state(self, state_name, state_value):
        self._cur_states[state_name] = state_value

    def register_updater(self, state_updater):
        self._state_updater = state_updater

    def compute_state(self, inputs):
        for input_name, input_value in inputs.items():
            if input_name not in self._inputs:
                raise ValueError('Unknown input %s. '
                                 'Please make sure %s in input '
                                 'place holder.' % (input_name, input_name))
            self._inputs[input_name] = input_value

        self._state_updater()

    def update_state(self):
        for _, decoder_state in self._states_holder.items():
            if self._cur_decoder_obj not in decoder_state:
                raise ValueError("Unknown decoder object, please make sure "
                                 "switch_decoder been invoked.")
            decoder_state[self._cur_decoder_obj].update_state(self._cur_states[
                state_name])


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

    @contextlib.contextmanager
    def block(self):
        if self._status != TrainingDecoder.BEFORE_DECODER:
            raise ValueError("decoder.block() can only be invoked once")
        self._status = TrainingDecoder.IN_DECODER
        with self._dynamic_rnn.block():
            self._state_cell.switch_decoder(self)
            yield
        self._status = TrainingDecoder.AFTER_DECODER

    @property
    def state_cell(self):
        self._assert_in_decoder_block("state_cell")
        return self._state_cell

    @property
    def dynamic_rnn(self):
        return self._dynamic_rnn

    @property
    def type(self):
        return self._type

    def step_input(self, x):
        self._assert_in_decoder_block("step_input")
        return self._dynamic_rnn.step_input(x)

    def static_input(self, x):
        self._assert_in_decoder_block("static_input")
        return self._dynamic_rnn.static_input(x)

    def __call__(self, *args, **kwargs):
        return self._dynamic_rnn(*args, **kwargs)

    def output(self, *outputs):
        self._assert_in_decoder_block("output")
        self._dynamic_rnn(output)

    def _assert_in_decoder_block(self, method):
        if self._status != TrainingDecoder.IN_DECODER:
            raise ValueError("%s should be invoked inside training "
                             "decoder." % method)


class BeamSearchDecoder(object):
    def __init__(self, state_cell):
        pass
