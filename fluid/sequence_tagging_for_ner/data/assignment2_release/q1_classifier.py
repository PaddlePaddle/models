import time
import math
import numpy as np
import tensorflow as tf
from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils import data_iterator

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  n_samples = 1024
  n_features = 100
  n_classes = 5
  # You may adjust the max_epochs to ensure convergence.
  max_epochs = 50
  # You may adjust this learning rate to ensure convergence.
  lr = 1e-4 

class SoftmaxModel(Model):
  """Implements a Softmax classifier with cross-entropy loss."""

  def load_data(self):
    """Creates a synthetic dataset and stores it in memory."""
    np.random.seed(1234)
    self.input_data = np.random.rand(
        self.config.n_samples, self.config.n_features)
    self.input_labels = np.ones((self.config.n_samples,), dtype=np.int32)

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (batch_size, n_features), type tf.float32
    labels_placeholder: Labels placeholder tensor of shape
                       (batch_size, n_classes), type tf.int32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

  def create_feed_dict(self, input_batch, label_batch):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    If label_batch is None, then no labels are added to feed_dict.

    Hint: The keys for the feed_dict should match the placeholder tensors
          created in add_placeholders.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return feed_dict

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return train_op

  def add_model(self, input_data):
    """Adds a linear-layer plus a softmax transformation

    The core transformation for this model which transforms a batch of input
    data into a batch of predictions. In this case, the mathematical
    transformation effected is

    y = softmax(xW + b)

    Hint: Make sure to create tf.Variables as needed. Also, make sure to use
          tf.name_scope to ensure that your name spaces are clean.
    Hint: For this simple use-case, it's sufficient to initialize both weights W
          and biases b with zeros.

    Args:
      input_data: A tensor of shape (batch_size, n_features).
    Returns:
      out: A tensor of shape (batch_size, n_classes)
    """
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return out

  def add_loss_op(self, pred):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: Use the cross_entropy_loss function we defined. This should be a very
          short function.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return loss

  def run_epoch(self, sess, input_data, input_labels):
    """Runs an epoch of training.

    Trains the model for one-epoch.
  
    Args:
      sess: tf.Session() object
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      average_loss: scalar. Average minibatch loss of model on epoch.
    """
    # And then after everything is built, start the training loop.
    average_loss = 0
    for step, (input_batch, label_batch) in enumerate(
        data_iterator(input_data, input_labels,
                      batch_size=self.config.batch_size,
                      label_size=self.config.n_classes)):

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = self.create_feed_dict(input_batch, label_batch)

      # Run one step of the model.  The return values are the activations
      # from the `self.train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
      average_loss += loss_value

    average_loss = average_loss / step
    return average_loss 

  def fit(self, sess, input_data, input_labels):
    """Fit model on provided data.

    Args:
      sess: tf.Session()
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      losses: list of loss per epoch
    """
    losses = []
    for epoch in range(self.config.max_epochs):
      start_time = time.time()
      average_loss = self.run_epoch(sess, input_data, input_labels)
      duration = time.time() - start_time
      # Print status to stdout.
      print('Epoch %d: loss = %.2f (%.3f sec)'
             % (epoch, average_loss, duration))
      losses.append(average_loss)
    return losses

  def __init__(self, config):
    """Initializes the model.

    Args:
      config: A model configuration object of type Config
    """
    self.config = config
    # Generate placeholders for the images and labels.
    self.load_data()
    self.add_placeholders()
    self.pred = self.add_model(self.input_placeholder)
    self.loss = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)
  
def test_SoftmaxModel():
  """Train softmax model for a number of steps."""
  config = Config()
  with tf.Graph().as_default():
    model = SoftmaxModel(config)
  
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
  
    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)
  
    losses = model.fit(sess, model.input_data, model.input_labels)

  # If ops are implemented correctly, the average loss should fall close to zero
  # rapidly.
  assert losses[-1] < .5
  print "Basic (non-exhaustive) classifier tests pass\n"

if __name__ == "__main__":
    test_SoftmaxModel()
