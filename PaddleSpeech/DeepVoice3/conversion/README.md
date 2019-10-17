# Parameter conversion

## generate name map

To convert a model trained with `https://github.com/r9y9/deepvoice3_pytorch`, we provide a script to generate name map between pytorch model and paddle model for `deepvoice3`. You can provide `--preset` and `--hparams` to specify the model's configuration.

```bash
python generate_name_map.py --preset=${preset_to_use} --haprams="hyper parameters to overwrite"
```

It would print a name map. The format of the name map file looks like this. Each line consists of 3 fields, the first is the name of a parameter in the saved state dict of pytorch model, the second and third is the name and shape of the corresponding parameter in the saved state dict of paddle.


```
seq2seq.encoder.embed_tokens.weight    encoder/Encoder_0/Embedding_0.w_0    [149, 256]
seq2seq.encoder.convolutions.0.bias    encoder/Encoder_0/ConvProj1D_1/Conv2D_0.b_0    [512]
seq2seq.encoder.convolutions.0.weight_g    encoder/Encoder_0/ConvProj1D_1/Conv2D_0.w_1    [512]
```

Redirect the output to a file to save it.

```bash
python generate_name_map.py --preset=${preset_to_use} --haprams="hyper parameters to overwrite" > name_map.txt
```

## convert saved pytorch model to paddle model

Given a name map and a saved pytorch model, you can convert it to a paddle model.

```bash
python convert.py \
    --pytorch-model ${pytorch_model.pth} \
    --paddle-model ${path_to_save_paddle_model} \
    --name-map ${name_map_path}
```

Note that the user should provide the name map file, and ensure the models are equivalent to each other and corresponding parameters have the right shapes.
