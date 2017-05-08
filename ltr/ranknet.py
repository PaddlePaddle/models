import os, sys
import gzip
import paddle.v2 as paddle
import numpy as np

# ranknet is the classic pairwise learning to rank algorithm
# http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf

def half_ranknet(name_prefix, input_dim):

  data = paddle.layer.data(name_prefix+"/data", paddle.data_type.dense_vector(input_dim))

  # two hidden layers
  hd1 = paddle.layer.fc(
    name=name_prefix+"/hidden_1",
    input=data,
    size=32,
    act=paddle.activation.Tanh(),
    param_attr=paddle.attr.Param(initial_std=0.01, name="hidden_w1"))
  hd2 = paddle.layer.fc(
    name=name_prefix+"/hidden_2",
    input=hd1,
    size=16,
    act=paddle.activation.Tanh(),
    param_attr=paddle.attr.Param(initial_std=0.01, name="hidden_w2"))
  output = paddle.layer.fc(
    name=name_prefix+"/output",
    input=hd2,
    size=1,
    act=paddle.activation.Linear(),
    param_attr=paddle.attr.Param(initial_std=0.01, name="output"))
  return output


def ranknet(input_dim):
  label = paddle.layer.data("label", paddle.data_type.integer_value(1))
  output_left = half_ranknet("left", input_dim)
  output_right = half_ranknet("right", input_dim)
  cost = paddle.layer.rank_cost(name="cost", left=output_left, right=output_right, label=label)
  return cost


def train_ranknet(num_passes):
  train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mq2007.train, buf_size=1000), batch_size=1000)
  test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mq2007.test, buf_size=1000), batch_size=1000)

  # mq2007 feature_dim = 46, dense format 
  # fc hidden_dim = 128
  feature_dim = 46
  cost = ranknet(feature_dim)
  parameters = paddle.parameters.create(cost)
  
  trainer = paddle.trainer.SGD(
    cost=cost,
    parameters=parameters,
    update_equation=paddle.optimizer.Adam(learning_rate=2e-4)
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
      with gzip.open("ranknet_params_%d.tar.gz" %(event.pass_id), "w") as f:
        parameters.to_tar(f)
  feeding = {"label":0,
             "left/data" :1,
             "right/data":2}
  trainer.train(reader=train_reader,
                event_handler=event_handler,
                feeding=feeding,
                num_passes=num_passes)

def ranknet_infer(pass_id):
  print "Begin to Infer..."
  feature_dim = 46
  output = half_ranknet("left", feature_dim)
  parameters = paddle.parameters.Parameters.from_tar(gzip.open("ranknet_params_%d.tar.gz" %(pass_id-1)))
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
  train_ranknet(2)
  ranknet_infer(pass_id=2)
