import paddle.v2 as paddle
from ntm_conf import gru_encoder_decoder
import gzip
import wmt14


def main():
    paddle.init(use_gpu=False, trainer_count=1)
    dict_size = 30000

    is_hybrid_addressing = True
    gen_creator = wmt14.gen(dict_size, src_seq_zero=is_hybrid_addressing)
    gen_data = []
    gen_num = 3

    for item in gen_creator():
        gen_data.append((item[0], item[1]))
        if len(gen_data) == gen_num:
            break

    beam_gen = gru_encoder_decoder(
        src_dict_dim=dict_size,
        trg_dict_dim=dict_size,
        is_generating=True,
        is_hybrid_addressing=is_hybrid_addressing)

    with gzip.open('./models/model_pass_00000.tar.gz') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    beam_result = paddle.infer(
        output_layer=beam_gen,
        parameters=parameters,
        input=gen_data,
        field=['prob', 'id'])

    src_dict, trg_dict = wmt14.get_dict(dict_size)
    seq_list = []
    seq = []
    for w in beam_result[1]:
        if w != -1:
            seq.append(w)
        else:
            seq_list.append(' '.join([trg_dict.get(w) for w in seq[1:]]))
            seq = []

    prob = beam_result[0]
    beam_size = 3
    for i in xrange(gen_num):
        print "\n*******************************************************\n"
        print "src:", ' '.join([src_dict.get(w) for w in gen_data[i][0]]), "\n"
        for j in xrange(beam_size):
            print "prob = %f:" % (prob[i][j]), seq_list[i * beam_size + 1]


if __name__ == '__main__':
    main()
