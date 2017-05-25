import paddle.v2 as paddle
from ntm_conf import gru_encoder_decoder
import wmt14
import sys
import gzip


def main():
    paddle.init(use_gpu=False, trainer_count=1, log_error_clipping=True)
    dict_size = 30000

    is_hybrid_addressing = True
    cost = gru_encoder_decoder(
        src_dict_dim=dict_size,
        trg_dict_dim=dict_size,
        is_generating=False,
        is_hybrid_addressing=is_hybrid_addressing)

    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-4,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=2500),
        learning_rate_decay_a=0.0,
        learning_rate_decay_b=0.0,
        gradient_clipping_threshold=25)

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # define data reader
    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            wmt14.train(dict_size, src_seq_zero=is_hybrid_addressing),
            buf_size=8192),
        batch_size=5)

    def event_handler(event):
        if isinstance(event, paddle.event.EndPass):
            model_name = './models/model_pass_%05d.tar.gz' % event.pass_id
            print('Save model to %s !' % model_name)
            with gzip.open(model_name, 'w') as f:
                parameters.to_tar(f)

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print("\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

            if event.batch_id % 100 == 0:
                model_name = './models/model_pass_%05d.tar.gz' % event.pass_id
                print('Save model to %s !' % model_name)
                with gzip.open(model_name, 'w') as f:
                    parameters.to_tar(f)

    if is_hybrid_addressing == True:
        feeding = {
            'source_language_word': 0,
            'init_attention_weights': 1,
            'target_language_word': 2,
            'target_language_next_word': 3
        }
    else:
        feeding = {
            'source_language_word': 0,
            'target_language_word': 1,
            'target_language_next_word': 2
        }

    # start to train
    trainer.train(
        reader=wmt14_reader,
        event_handler=event_handler,
        num_passes=2,
        feeding=feeding)


if __name__ == '__main__':
    main()
