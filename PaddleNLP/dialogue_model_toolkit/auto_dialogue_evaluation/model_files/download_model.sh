#matching pretrained
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/auto_dialogue_evaluation_matching_pretrained-1.0.0.tar.gz
tar -xzf auto_dialogue_evaluation_matching_pretrained-1.0.0.tar.gz

#finetuned
for task in seq2seq_naive seq2seq_att keywords human
do
  wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/auto_dialogue_evaluation_${task}_finetuned-1.0.0.tar.gz
  tar -xzf auto_dialogue_evaluation_${task}_finetuned-1.0.0.tar.gz
done

