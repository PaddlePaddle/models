import os, sys
import gzip
import paddle.v2 as paddle
import numpy as np
import functools

#lambdaRank is listwise learning to rank algorithm

def lambdaRank(feature_dim):
  label = paddle.layer.data("label", paddle.data_type.integer_value_sequence(1))
  data = paddle.layer.data("data", paddle.data_type.dense_vector(feature_dim))

  # two hidden layers
  hd1 = paddle.layer.fc(
    name="/hidden_1",
    input=data,
    size=256,
    act=paddle.activation.Tanh(),
    param_attr=paddle.attr.Param(initial_std=0.01, name="hidden_w1"))
  hd2 = paddle.layer.fc(
    name="/hidden_2",
    input=hd1,
    size=256,
    act=paddle.activation.Tanh(),
    param_attr=paddle.attr.Param(initial_std=0.01, name="hidden_w2"))
  output = paddle.layer.fc(
    name="/output",
    input=hd2,
    size=1,
    act=paddle.activation.Linear(),
    param_attr=paddle.attr.Param(initial_std=0.01, name="output"))
  cost = paddle.layer.lambda_cost(input=output,
                                  score=label,
                                  NDCG_num=10)
  return cost, output
  

def train_lambdaRank(num_passes):
  fill_default_train = functools.partial(paddle.dataset.mq2007.train, format="listwise")
  fill_default_test = functools.partial(paddle.dataset.mq2007.test, format="listwise")
  train_reader = paddle.batch(
    paddle.reader.shuffle(fill_default_train, buf_size=1000), batch_size=1000)
  test_reader = paddle.batch(
    paddle.reader.shuffle(fill_default_test, buf_size=1000), batch_size=1000)

  # mq2007 feature_dim = 46, dense format 
  # fc hidden_dim = 128
  feature_dim = 46
  cost, output = lambdaRank(feature_dim)
  parameters = paddle.parameters.create(cost)
  
  trainer = paddle.trainer.SGD(
    cost=cost,
    parameters=parameters,
    update_equation=paddle.optimizer.Adam(learning_rate=1e-4)
    )

  #  Define end batch and end pass event handler
  def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
      if event.batch_id % 100 == 0:
        print "Pass %d Batch %d Cost %.9f" % (
          event.pass_id, event.batch_id, event.cost)
      else:
        sys.stdout.write(".")
        sys.stdout.flush()
    if isinstance(event, paddle.event.EndPass):
      result = trainer.test(reader=test_reader, feeding=feeding)
      print "\nTest with Pass %d, %s" %(event.pass_id, result.metrics)
      with gzip.open("lambdaRank_params_%d.tar.gz" %(event.pass_id), "w") as f:
        parameters.to_tar(f)
  feeding = {"label":0,
             "data": 1}
  trainer.train(reader=train_reader,
                event_handler=event_handler,
                feeding=feeding,
                num_passes=num_passes)

def lambdaRank_infer(pass_id):
  print "Begin to Infer..."
  feature_dim = 46
  output = lambdaRnak(feature_dim)
  parameters = paddle.parameters.Parameters.from_tar(gzip.open("lambdaRank_params_%d.tar.gz" %(pass_id-1)))
  infer_data = []
  infer_data_num = 1000
  for label, left, right in paddle.dataset.mq2007.test():
    infer_data.append(left)
    if len(infer_data) == infer_data_num:
      break
  predicitons = paddle.infer(output_layer=output,
                             parameters=parameters,
                             input=infer_data)
  for i, score in enumerate(predicitons):
    print score

if __name__ == '__main__':
  paddle.init(use_gpu=False, trainer_count=4)
  train_lambdaRank(2)
  lambdaRank_infer(pass_id=2)
