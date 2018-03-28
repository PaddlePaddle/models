import math

__all__ = ["TrainingConfig", "InferConfig"]


class CommonConfig(object):
    def __init__(self):
        # network size:
        # dimension of the question LSTM
        self.q_lstm_dim = 64
        # dimension of the attention layer
        self.latent_chain_dim = 64
        # dimension of the evidence LSTMs
        self.e_lstm_dim = 64
        # dimension of the qe.comm and ee.comm feature embeddings
        self.com_vec_dim = 2
        self.drop_rate = 0.05

        # CRF:
        # valid values are BIO and BIO2
        self.label_schema = "BIO2"

        # word embedding:
        # vocabulary file path
        self.word_dict_path = "data/embedding/wordvecs.vcb"
        # word embedding file path
        self.wordvecs_path = "data/embedding/wordvecs.txt"
        self.word_vec_dim = 64

        # saving model & logs:
        # dir for saving models
        self.model_save_dir = "models"

        # print training info every log_period batches
        self.log_period = 100
        # show parameter status every show_parameter_status_period batches
        self.show_parameter_status_period = 100

    @property
    def label_num(self):
        if self.label_schema == "BIO":
            return 3
        elif self.label_schema == "BIO2":
            return 4
        else:
            raise ValueError("wrong value for label_schema")

    @property
    def default_init_std(self):
        return 1 / math.sqrt(self.e_lstm_dim * 4)

    @property
    def default_l2_rate(self):
        return 8e-4 * self.batch_size / 6

    @property
    def dict_dim(self):
        return len(self.vocab)


class TrainingConfig(CommonConfig):
    def __init__(self):
        super(TrainingConfig, self).__init__()

        # data:
        # training data path
        self.train_data_path = "data/data/training.json.gz"

        # number of batches used in each pass
        self.batches_per_pass = 1000
        # number of passes to train
        self.num_passes = 25
        # batch size
        self.batch_size = 120

        # the ratio of negative samples used in training
        self.negative_sample_ratio = 0.2
        # the ratio of negative samples that contain golden answer string
        self.hit_ans_negative_sample_ratio = 0.25

        # keep only first B in golden labels
        self.keep_first_b = False

        # use GPU to train the model
        self.use_gpu = False
        # number of threads
        self.trainer_count = 1

        # random seeds:
        # data reader random seed, 0 for random seed
        self.seed = 0
        # paddle random seed, 0 for random seed
        self.paddle_seed = 0

        # optimizer:
        self.learning_rate = 1e-3
        # rmsprop
        self.rho = 0.95
        self.epsilon = 1e-4
        # model average
        self.average_window = 0.5
        self.max_average_window = 10000


class InferConfig(CommonConfig):
    def __init__(self):
        super(InferConfig, self).__init__()

        self.use_gpu = False
        self.trainer_count = 1
        self.batch_size = 120
        self.wordvecs = None
