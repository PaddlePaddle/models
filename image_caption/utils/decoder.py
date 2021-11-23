import paddle.fluid as fluid
import paddle.fluid.layers as pd
# from .config import config

target_dict_dim = 4528
decoder_size = 64

def decode(context,trg_embedding):

    rnn = pd.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        pre_state = rnn.memory(init=context, need_reorder=True)
        current_state = pd.fc(
            input=[current_word, pre_state], size=decoder_size, act='tanh')

        current_score = pd.fc(
            input=current_state, size=target_dict_dim, act='softmax')
        rnn.update_memory(pre_state, current_state)
        rnn.output(current_score)

    return rnn()