# paddlenlp.metrics

## Perplexity
Perplexity is calculated using cross entropy. It supports both padding data
and no padding data.

If data is not padded, users should provide `seq_len` for `Metric`
initialization. If data is padded, your label should contain `seq_mask`,
which indicates the actual length of samples.

This Perplexity requires that the output of your network is prediction,
label and sequence length (opitonal). If the Perplexity here doesn't meet
your needs, you could override the `compute` or `update` method for
caculating Perplexity.

## BLEU
BLEU (bilingual evaluation understudy) is an algorithm for evaluating the
quality of text which has been machine-translated from one natural language
to another. This metric uses a modified form of precision to compare a
candidate translation against multiple reference translations.

BLEU could be used as `paddle.metrics.Metric` class, or an ordinary
class.

When BLEU is used as `paddle.metrics.Metric` class. A function is
needed that transforms the network output to reference string list, and
transforms the label to candidate string. By default, a default function
`_default_trans_func` is provided, which gets target sequence id by
calculating the maximum probability of each step. In this case, user must
provide `vocab`. It should be noted that the BLEU here is different from
the BLEU calculated in prediction, and it is only for observation during
training and evaluation.

## Rouge
### rouge-l

## dureader
## chunk
## squad
