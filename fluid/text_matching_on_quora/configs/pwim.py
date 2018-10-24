
import basic_config

def pwim_base():
    """
    set configs
    """
    config = basic_config.config()
    config.batch_size = 128
    config.learning_rate = 0.001
    config.save_dirname = "pwim_model"
    #config.use_pretrained_word_embedding = False
    config.use_pretrained_word_embedding = True
    config.dict_dim = 40000 # approx_vocab_size
    #config.use_cuda = False
    config.use_cuda = True

    
    # net config
    config.emb_dim = 300
    config.kernel_size = 5
    config.kernel_count = 300
    config.fc_dim = 128
    config.mlp_hid_dim = [128, 128]
    config.droprate_conv = 0.1
    config.droprate_fc = 0.1
    config.class_dim = 2
    config.seq_limit_len = 48

    return config 
