
import basic_config

def sse_base():
    """
    use config in the paper 'Shortcut-Stacked Sentence Encoders for Multi-Domain Inference'
    """
    config = basic_config.config()
    config.learning_rate = 0.0002
    config.lr_decay = 0.7
    config.save_dirname = "model_dir"
    config.use_pretrained_word_embedding = True
    config.dict_dim = 40000 # approx_vocab_size
    config.metric_type = ['accuracy']
    config.optimizer_type = 'adam'
    config.use_lod_tensor = True
    config.embedding_norm = False
    config.OOV_fill = 'uniform'
    config.duplicate_data = False
    
    # net config
    config.emb_dim = 300
    config.rnn_hid_dim = [512, 1024, 2048]
    config.fc_dim = [1600, 1600]
    config.droprate_lstm = 0.0
    config.droprate_fc = 0.1
    config.class_dim = 2

    return config
