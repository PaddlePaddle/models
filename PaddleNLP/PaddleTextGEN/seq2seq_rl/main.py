import numpy as np
import paddle.fluid as fluid
from pg_agent import SeqPGAgent
from model import Seq2SeqModel, PolicyGradient, reward_func
import reader
from args import parse_args

def main():
    args = parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    dropout_prob = args.dropout
    src_vocab_size = args.src_vocab_size
    trg_vocab_size = args.tar_vocab_size
    batch_size = args.batch_size
    lr = args.learning_rate
    train_data_prefix = args.train_data_prefix
    eval_data_prefix = args.eval_data_prefix
    test_data_prefix = args.test_data_prefix
    vocab_prefix = args.vocab_prefix
    src_lang = args.src_lang
    tar_lang = args.tar_lang
    print("begin to load data")
    raw_data = reader.raw_data(src_lang, tar_lang, vocab_prefix,
                               train_data_prefix, eval_data_prefix,
                               test_data_prefix, args.max_len)
    print("finished load data")
    train_data, valid_data, test_data, _ = raw_data

    def prepare_input(batch, epoch_id=0, teacher_forcing=False):
        src_ids, src_mask, tar_ids, tar_mask = batch
        res = {}
        src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1]))
        in_tar = tar_ids[:, :-1]
        label_tar = tar_ids[:, 1:]

        in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1]))
        label_tar = label_tar.reshape(
            (label_tar.shape[0], label_tar.shape[1], 1))

        res['src'] = src_ids
        res['src_sequence_length'] = src_mask
        if teacher_forcing:
            res['tar'] = in_tar
            res['tar_sequence_length'] = tar_mask
            res['label'] = label_tar

        return res, np.sum(tar_mask)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    ### TODO: pretrain the model with teacher-forcing MLE

    ### fine-tune with policy gradient
    agent = SeqPGAgent(model_cls=Seq2SeqModel,
                       alg_cls=PolicyGradient,
                       reward_func=reward_func,
                       model_hparams={
                           "num_layers": num_layers,
                           "hidden_size": hidden_size,
                           "dropout_prob": dropout_prob,
                           "src_vocab_size": src_vocab_size,
                           "trg_vocab_size": trg_vocab_size,
                           "bos_token": 1,
                           "eos_token": 2,
                       },
                       alg_hparams={"lr": lr},
                       executor=exe)

    exe.run(fluid.default_startup_program())
    if args.reload_model:  # load MLE pre-trained model
        fluid.io.load_params(exe,
                             dirname=args.reload_model,
                             main_program=agent.full_program)

    max_epoch = args.max_epoch
    for epoch_id in range(max_epoch):
        if args.enable_ce:
            train_data_iter = reader.get_data_iter(
                train_data, batch_size, enable_ce=True)
        else:
            train_data_iter = reader.get_data_iter(train_data, batch_size)
        for batch_id, batch in enumerate(train_data_iter):
            input_data_feed, word_num = prepare_input(batch, epoch_id=epoch_id)
            reward, cost = agent.learn(input_data_feed)
            print("epoch_id: %d, batch_id: %d, reward: %f, cost: %f" %
                  (epoch_id, batch_id, reward.mean(), cost))


if __name__ == '__main__':
    main()