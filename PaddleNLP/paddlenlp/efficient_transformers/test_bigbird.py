import paddle
import numpy as np
from paddlenlp.efficient_transformers import TransformerEncoderLayer, TransformerEncoder
import unittest
import paddle.fluid as fluid


class TestBigBird(unittest.TestCase):
    def init_device(self):
        paddle.set_device("cpu")

    def init_params(self):
        pass

    def setUp(self):
        self.init_device()
        self.seed = 102
        self.src_len = 20
        self.batch_size = 2
        self.d_model = 128
        self.num_rand_blocks = 2
        self.encoder_layers = 1
        self.initializer_const = 0.01
        self.nhead = 2
        self.dim_feedforward = 512
        self.block_size = 1
        self.window_size = 3
        self.num_global_blocks = 1
        self.init_params()

        self.weight_attr = [
            fluid.initializer.Constant(value=self.initializer_const),
            fluid.initializer.Constant(value=self.initializer_const)
        ]
        self.bias_attr = [
            fluid.initializer.Constant(value=self.initializer_const),
            fluid.initializer.Constant(value=self.initializer_const)
        ]
        self.input = paddle.rand((self.batch_size, self.src_len, self.d_model))

        bigbird_kwargs = {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": 0.0,
            "activation": "relu",
            "normalize_before": False,
            "weight_attr": self.weight_attr,
            "bias_attr": self.bias_attr,
            "attention_type": "bigbird",
            "seed": self.seed,
            "block_size": self.block_size,
            "window_size": self.window_size,
            "num_global_blocks": self.num_global_blocks,
            "num_rand_blocks": self.num_rand_blocks
        }

        bigbird_simulated_kwargs = dict(bigbird_kwargs)
        bigbird_simulated_kwargs["attention_type"] = "bigbird_simulated"

        bigbird_layer = TransformerEncoderLayer(**bigbird_kwargs)
        bigbird_simulated_layer = TransformerEncoderLayer(
            **bigbird_simulated_kwargs)

        self.bigbird_encoder = TransformerEncoder(bigbird_layer,
                                                  self.encoder_layers)
        self.bigbird_simulated_encoder = TransformerEncoder(
            bigbird_simulated_layer, self.encoder_layers)

    def test_equal(self):
        bigbird_out = self.bigbird_encoder(self.input)
        bigbird_simulated_out = self.bigbird_simulated_encoder(self.input)
        self.assertTrue(
            np.allclose(
                bigbird_out.numpy(), bigbird_simulated_out.numpy(), atol=1e-6))


class TestBigBirdGPU(TestBigBird):
    def init_device(self):
        paddle.set_device("gpu:0")


class TestBigBirdMultiLayers1(TestBigBird):
    def init_params(self):
        self.encoder_layers = 2


class TestBigBirdMultiLayers2(TestBigBird):
    def init_params(self):
        self.encoder_layers = 5


class TestBigBirdWindowSize1(TestBigBird):
    def init_params(self):
        self.window_size = 5


class TestBigBirdBlockSize1(TestBigBird):
    def init_params(self):
        self.src_len = 20
        self.block_size = 2


class TestBigBirdBlockSize2(TestBigBird):
    def init_params(self):
        self.src_len = 45
        self.block_size = 3


class TestBigBirdLongSeq(TestBigBird):
    def init_params(self):
        self.src_len = 4096
        self.block_size = 64


class TestBigBirdMultiHead(TestBigBird):
    def init_params(self):
        self.nhead = 8


class TestBigBirdMultiGlobalBlocks(TestBigBird):
    def init_params(self):
        self.src_len = 256
        self.block_size = 4
        self.num_global_blocks = 3


if __name__ == '__main__':
    unittest.main()
