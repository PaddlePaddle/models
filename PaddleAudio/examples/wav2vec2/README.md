# Wav2vec 2.0 for Speech Recognition

This is paddlepaddle version of Facebook's Wav2vec 2.0 [1], with code and pre-trained weighted ported from [Fairseq](https://github.com/pytorch/fairseq/) and [huggingface](https://github.com/huggingface/transformers).

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
python test.py --device "gpu" --audio <audio_file> --config <config_name>
```
If you do not have gpu or run out of gpu memory, try cpu:
``` bash
python test.py --device "cpu" --audio <audio_file> --config <config_name>
```

Supported config-names are  "wav2vec2-base-960h", "wav2vec2-large-960h", "wav2vec2-large-960h-lv60", "wav2vec2-large-960h-lv60-self".

If no audio file is provided, a sample audio will be loaded.
If successful, you will see output as follows,
```
[2021-06-07 20:43:27,783] [INFO] - pred==> particularly so on this last night when only two of the little cubicles were occupied the thousands of others standing with dark empty doors
[2021-06-07 20:43:27,783] [INFO] - true==> particularly so on this last night when only two of the little cubicles were occupied the thousands of others standing with dark empty doors
```

## Accuracy notes
The accuracy test against origin implementation is provided in the [unit_test](../../test) section. Accuracy is measured in logit level by computing logic mean and standard-deviation and the discrepancy is less than 1e-4 for both mean and std.

## Performance

Performance for all four models is measured in word error rate (WER). Librispeech test and dev splits are used. To simplify this example, we do not incorporate language models (LM).  Experiments results are summarized in the following table.

|model-name|LM|dev-clean |dev-other |test-clean |test-other |
| :--- | :--- | :--- | :--- |:--- |:--- |
|wav2vec2-base-960h|None| 3.7 |9.7 | 3.8 | 10.1 |
|wav2vec2-large-960h|None| 3.2 |7.3 | 3.2 | 7.7 |
|wav2vec2-large-960h-lv60|None| 2.6 |5.1 | 2.6 | 5.6 |
|wav2vec2-large-960h-lv60-self|None| 2.1 |4.3 | 2.3 | 5.0 |

 For the largest model "wav2vec2-large-960h-lv60-self" with self-training but without LM, the WER for test-clean and dev-clean are 2.3 and 2.1 respectively, which are very close to the results in the paper (2.0/1.9).

 Users can replicate the above results by
 ```
python test_librispeech.py -d <device> -t <test-dev-clean-other-path> -c <model-name> -o <result_file.txt>
 ```


## Reference
[1] Baevski, Alexei, et al. “Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 12449–12460.
