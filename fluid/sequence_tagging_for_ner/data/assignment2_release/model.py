class Model(object):
  """Abstracts a Tensorflow graph for a learning task.

  We use various Model classes as usual abstractions to encapsulate tensorflow
  computational graphs. Each algorithm you will construct in this homework will
  inherit from a Model object.
  """

  def load_data(self):
    """Loads data from disk and stores it in memory.

    Feel free to add instance variables to Model object that store loaded data.    
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_placeholders(self):
    """Adds placeholder variables to tensorflow computational graph.

    Tensorflow uses placeholder variables to represent locations in a
    computational graph where data is inserted.  These placeholders are used as
    inputs by the rest of the model building code and will be fed data during
    training.

    See for more information:

    https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def create_feed_dict(self, input_batch, label_batch):
    """Creates the feed_dict for training the given step.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
  
    If label_batch is None, then no labels are added to feed_dict.

    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_model(self, input_data):
    """Implements core of model that transforms input_data into predictions.

    The core transformation for this model which transforms a batch of input
    data into a batch of predictions.

    Args:
      input_data: A tensor of shape (batch_size, n_features).
    Returns:
      out: A tensor of shape (batch_size, n_classes)
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_loss_op(self, pred):
    """Adds ops for loss to the computational graph.

    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar) output
    """
    raise NotImplementedError("Each Model must re-implement this method.")

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
    raise NotImplementedError("Each Model must re-implement this method.")

  def fit(self, sess, input_data, input_labels):
    """Fit model on provided data.

    Args:
      sess: tf.Session()
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      losses: list of loss per epoch
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def predict(self, sess, input_data, input_labels=None):
    """Make predictions from the provided model.
    Args:
      sess: tf.Session()
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      average_loss: Average loss of model.
      predictions: Predictions of model on input_data
    """
    raise NotImplementedError("Each Model must re-implement this method.")

class LanguageModel(Model):
  """Abstracts a Tensorflow graph for learning language models.

  Adds ability to do embedding.
  """

  def add_embedding(self):
    """Add embedding layer. that maps from vocabulary to vectors.
    """
    raise NotImplementedError("Each Model must re-implement this method.")
