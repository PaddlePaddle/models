class TrainTaskConfig(object):
    use_gpu = True
    # the epoch number to train.
    pass_num = 30
    # the number of sequences contained in a mini-batch.
    batch_size = 32
    # the hyper parameters for Adam optimizer.
    # This static learning_rate will be multiplied to the LearningRateScheduler
    # derived learning rate the to get the final learning rate.
    learning_rate = 1
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-9
    # the parameters for learning rate scheduling.
    warmup_steps = 4000
    # the flag indicating to use average loss or sum loss when training.
    use_avg_cost = True
    # the weight used to mix up the ground-truth distribution and the fixed
    # uniform distribution in label smoothing when training.
    # Set this as zero if label smoothing is not wanted.
    label_smooth_eps = 0.1
    # the directory for saving trained models.
    model_dir = "trained_models"
    # the directory for saving checkpoints.
    ckpt_dir = "trained_ckpts"
    # the directory for loading checkpoint.
    # If provided, continue training from the checkpoint.
    ckpt_path = None
    # the parameter to initialize the learning rate scheduler.
    # It should be provided if use checkpoints, since the checkpoint doesn't
    # include the training step counter currently.
    start_step = 0


class InferTaskConfig(object):
    use_gpu = True
    # the number of examples in one run for sequence generation.
    batch_size = 10
    # the parameters for beam search.
    beam_size = 5
    max_length = 30
    # the number of decoded sentences to output.
    n_best = 1
    # the flags indicating whether to output the special tokens.
    output_bos = False
    output_eos = False
    output_unk = False
    # the directory for loading the trained model.
    model_path = "trained_models/pass_1.infer.model"


class ModelHyperParams(object):
    # This model directly uses paddle.dataset.wmt16 in which <bos>, <eos> and
    # <unk> token has alreay been added. As for the <pad> token, any token
    # included in dict can be used to pad, since the paddings' loss will be
    # masked out and make no effect on parameter gradients.
    # size of source word dictionary.
    src_vocab_size = 10000
    # size of target word dictionay
    trg_vocab_size = 10000
    # index for <bos> token
    bos_idx = 0
    # index for <eos> token
    eos_idx = 1
    # index for <unk> token
    unk_idx = 2
    # max length of sequences.
    # The size of position encoding table should at least plus 1, since the
    # sinusoid position encoding starts from 1 and 0 can be used as the padding
    # token for position encoding.
    max_length = 50
    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.
    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 1024
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    n_layer = 6
    # dropout rate used by all dropout layers.
    dropout = 0.1


def merge_cfg_from_list(cfg_list, g_cfgs):
    """
    Set the above global configurations using the cfg_list. 
    """
    assert len(cfg_list) % 2 == 0
    for key, value in zip(cfg_list[0::2], cfg_list[1::2]):
        for g_cfg in g_cfgs:
            if hasattr(g_cfg, key):
                try:
                    value = eval(value)
                except SyntaxError:  # for file path
                    pass
                setattr(g_cfg, key, value)
                break


# Here list the data shapes and data types of all inputs.
# The shapes here act as placeholder and are set to pass the infer-shape in
# compile time.
input_descs = {
    # The actual data shape of src_word is:
    # [batch_size * max_src_len_in_batch, 1]
    "src_word": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    # The actual data shape of src_pos is:
    # [batch_size * max_src_len_in_batch, 1]
    "src_pos": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    # This input is used to remove attention weights on paddings in the
    # encoder.
    # The actual data shape of src_slf_attn_bias is:
    # [batch_size, n_head, max_src_len_in_batch, max_src_len_in_batch]
    "src_slf_attn_bias":
    [(1, ModelHyperParams.n_head, (ModelHyperParams.max_length + 1),
      (ModelHyperParams.max_length + 1)), "float32"],
    # This shape input is used to reshape the output of embedding layer.
    "src_data_shape": [(3L, ), "int32"],
    # This shape input is used to reshape before softmax in self attention.
    "src_slf_attn_pre_softmax_shape": [(2L, ), "int32"],
    # This shape input is used to reshape after softmax in self attention.
    "src_slf_attn_post_softmax_shape": [(4L, ), "int32"],
    # The actual data shape of trg_word is:
    # [batch_size * max_trg_len_in_batch, 1]
    "trg_word": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    # The actual data shape of trg_pos is:
    # [batch_size * max_trg_len_in_batch, 1]
    "trg_pos": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    # This input is used to remove attention weights on paddings and
    # subsequent words in the decoder.
    # The actual data shape of trg_slf_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_trg_len_in_batch]
    "trg_slf_attn_bias": [(1, ModelHyperParams.n_head,
                           (ModelHyperParams.max_length + 1),
                           (ModelHyperParams.max_length + 1)), "float32"],
    # This input is used to remove attention weights on paddings of the source
    # input in the encoder-decoder attention.
    # The actual data shape of trg_src_attn_bias is:
    # [batch_size, n_head, max_trg_len_in_batch, max_src_len_in_batch]
    "trg_src_attn_bias": [(1, ModelHyperParams.n_head,
                           (ModelHyperParams.max_length + 1),
                           (ModelHyperParams.max_length + 1)), "float32"],
    # This shape input is used to reshape the output of embedding layer.
    "trg_data_shape": [(3L, ), "int32"],
    # This shape input is used to reshape before softmax in self attention.
    "trg_slf_attn_pre_softmax_shape": [(2L, ), "int32"],
    # This shape input is used to reshape after softmax in self attention.
    "trg_slf_attn_post_softmax_shape": [(4L, ), "int32"],
    # This shape input is used to reshape before softmax in encoder-decoder
    # attention.
    "trg_src_attn_pre_softmax_shape": [(2L, ), "int32"],
    # This shape input is used to reshape after softmax in encoder-decoder
    # attention.
    "trg_src_attn_post_softmax_shape": [(4L, ), "int32"],
    # This input is used in independent decoder program for inference.
    # The actual data shape of enc_output is:
    # [batch_size, max_src_len_in_batch, d_model]
    "enc_output": [(1, (ModelHyperParams.max_length + 1),
                    ModelHyperParams.d_model), "float32"],
    # The actual data shape of label_word is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_word": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    # This input is used to mask out the loss of paddding tokens.
    # The actual data shape of label_weight is:
    # [batch_size * max_trg_len_in_batch, 1]
    "lbl_weight": [(1 * (ModelHyperParams.max_length + 1), 1L), "float32"],
}

# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )
# separated inputs for different usages.
encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias", )
encoder_util_input_fields = (
    "src_data_shape",
    "src_slf_attn_pre_softmax_shape",
    "src_slf_attn_post_softmax_shape", )
decoder_data_input_fields = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "enc_output", )
decoder_util_input_fields = (
    "trg_data_shape",
    "trg_slf_attn_pre_softmax_shape",
    "trg_slf_attn_post_softmax_shape",
    "trg_src_attn_pre_softmax_shape",
    "trg_src_attn_post_softmax_shape", )
label_data_input_fields = (
    "lbl_word",
    "lbl_weight", )
