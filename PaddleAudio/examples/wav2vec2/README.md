# Wav2vec2 models for Speech recognition

This is paddle-paddle version of Facebook's Wav2vec2.0 [1], with code and pre-trained weighted ported from [Fairseq](https://github.com/pytorch/fairseq/) and [huggingface](https://github.com/huggingface/transformers).

## Supported configs

|name|Finetuning split| Dataset |
| :--- | :--- | :---  |
|wav2vec2-base-960h|960h| [Librispeech](http://www.openslr.org/12)|
|wav2vec2-large-960h|960h| [Librispeech](http://www.openslr.org/12)|
|wav2vec2-base-960h-lv60|960h| [Librispeech](http://www.openslr.org/12) + [Libri-Light](https://github.com/facebookresearch/libri-light)|
|wav2vec2-base-960h-lv60-self|960h| [Librispeech](http://www.openslr.org/12) + [Libri-Light](https://github.com/facebookresearch/libri-light) + Self Training |

## Quickstart

Run the speech recognition test with your audio file,
``` bash
python test.py --device "gpu:0" --audio <audio_file> --config <config_name>
```
If you do not have gpu or run out of gpu memory, try cpu:
``` bash
python test.py --device "cpu:0" --audio <audio_file> --config <config_name>
```

Supported config-names are  "wav2vec2-base-960h", "wav2vec2-large-960h", "wav2vec2-large-960h-lv60", "wav2vec2-large-960h-lv60-self".

If no audio file is provided, a sample audio will be loaded.
If successful, you will see output as follows,
```
[2021-06-07 20:43:27,783] [INFO] - pred==> particularly so on this last night when only two of the little cubicles were occupied the thousands of others standing with dark empty doors
[2021-06-07 20:43:27,783] [INFO] - true==> particularly so on this last night when only two of the little cubicles were occupied the thousands of others standing with dark empty doors
```



## Reference
[1] Baevski, Alexei, et al. “Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 12449–12460.
