import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as pd
from utils import decoder,encoder,reader

from config import config


dict_size = config.dict_size
source_dict_dim = target_dict_dim = dict_size
hidden_dim = config.hidden_dim
word_dim = config.word_dim
batch_size = config.batch_size
max_length = 8
topk_size = 50
beam_size = 2
is_sparse = True
EPOCH_NUM = config.EPOCH_NUM
decoder_size = hidden_dim

model_save_dir = "image_caption.inference.model"


def encode():

    image = pd.data(
        name="image", shape=config.shape, dtype='float32')
    out = encoder.resnet(image)
    return out


def train_decoder(context):


    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=fluid.ParamAttr(name='vemb'))

    return decoder.decode(context,trg_embedding)


def train_program():
    context = encode()
    rnn_out = train_decoder(context)
    label = pd.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = pd.cross_entropy(input=rnn_out, label=label)
    avg_cost = pd.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.Adagrad(
        learning_rate=1e-3,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.1))


def train(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train(), buf_size=1000),
        batch_size=batch_size)
    feed_order = [
        'image','target_language_word','target_language_next_word'
    ]


    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):

            if event.epoch % 5 == 0 and event.step % 100 == 0:
                outs = trainer.test(reader=train_reader,
                feed_order=feed_order)
                avg_cost = outs[0]
                with open('cost.txt','a+') as f:
                    f.write('epoch:'+str(event.epoch)+'\t'+'cost:'+str(avg_cost)+'\n')

                print('pass_id=' + str(event.epoch) + ' batch=' + str(
                    event.step) + ' avg_cost=' + str(avg_cost))


            else:
                if event.step % 30 == 0:
                    print('pass_id=' + str(event.epoch) + ' batch=' + str(
                    event.step))


        if isinstance(event, fluid.EndEpochEvent):
            if event.epoch % 5 == 0:
                trainer.save_params(model_save_dir)

    trainer = fluid.Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_func)

    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=feed_order)
    

def main(use_cuda):
    train(use_cuda)


if __name__ == '__main__':
    use_cuda = False # set to True if training with GPU
    main(use_cuda)
