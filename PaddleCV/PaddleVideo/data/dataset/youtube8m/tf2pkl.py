#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""Provides readers configured for different datasets."""
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import logging
try:
    import cPickle as pickle
except:
    import pickle

from tensorflow.python.platform import gfile

assert (len(sys.argv) == 3)
source_dir = sys.argv[1]
target_dir = sys.argv[2]


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.

    Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

    Returns:
    A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def resize_axis(tensor, axis, new_size, fill_value=0):
    """Truncates or pads a tensor to new_size on on a given axis.

    Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
    size increases, the padding will be performed at the end, using fill_value.

    Args:
      tensor: The tensor to be resized.
      axis: An integer representing the dimension to be sliced.
      new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
      fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.

    Returns:
      The resized tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))

    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)

    resized = tf.concat([
        tf.slice(tensor, tf.zeros_like(shape), shape),
        tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized


class BaseReader(object):
    """Inherit from this class when implementing new readers."""

    def prepare_reader(self, unused_filename_queue):
        """Create a thread for generating prediction and label tensors."""
        raise NotImplementedError()


class YT8MFrameFeatureReader(BaseReader):
    """Reads TFRecords of SequenceExamples.

    The TFRecords must contain SequenceExamples with the sparse in64 'labels'
    context feature and a fixed length byte-quantized feature vector, obtained
    from the features in 'feature_names'. The quantized features will be mapped
    back into a range between min_quantized_value and max_quantized_value.
    """

    def __init__(self,
                 num_classes=3862,
                 feature_sizes=[1024],
                 feature_names=["inc3"],
                 max_frames=300):
        """Construct a YT8MFrameFeatureReader.

        Args:
          num_classes: a positive integer for the number of classes.
          feature_sizes: positive integer(s) for the feature dimensions as a list.
          feature_names: the feature name(s) in the tensorflow record as a list.
          max_frames: the maximum number of frames to process.
        """

        assert len(feature_names) == len(feature_sizes), \
        "length of feature_names (={}) != length of feature_sizes (={})".format( \
        len(feature_names), len(feature_sizes))

        self.num_classes = num_classes
        self.feature_sizes = feature_sizes
        self.feature_names = feature_names
        self.max_frames = max_frames

    def get_video_matrix(self, features, feature_size, max_frames,
                         max_quantized_value, min_quantized_value):
        """Decodes features from an input string and quantizes it.

        Args:
          features: raw feature values
          feature_size: length of each frame feature vector
          max_frames: number of frames (rows) in the output feature_matrix
          max_quantized_value: the maximum of the quantized value.
          min_quantized_value: the minimum of the quantized value.

        Returns:
          feature_matrix: matrix of all frame-features
          num_frames: number of frames in the sequence
        """
        decoded_features = tf.reshape(
            tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
            [-1, feature_size])

        num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)

        feature_matrix = decoded_features

        return feature_matrix, num_frames

    def prepare_reader(self,
                       filename_queue,
                       max_quantized_value=2,
                       min_quantized_value=-2):
        """Creates a single reader thread for YouTube8M SequenceExamples.

        Args:
          filename_queue: A tensorflow queue of filename locations.
          max_quantized_value: the maximum of the quantized value.
          min_quantized_value: the minimum of the quantized value.

        Returns:
          A tuple of video indexes, video features, labels, and padding data.
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        contexts, features = tf.parse_single_sequence_example(
            serialized_example,
            context_features={
                "id": tf.FixedLenFeature([], tf.string),
                "labels": tf.VarLenFeature(tf.int64)
            },
            sequence_features={
                feature_name: tf.FixedLenSequenceFeature(
                    [], dtype=tf.string)
                for feature_name in self.feature_names
            })

        # read ground truth labels
        labels = (tf.cast(
            tf.sparse_to_dense(
                contexts["labels"].values, (self.num_classes, ),
                1,
                validate_indices=False),
            tf.bool))

        # loads (potentially) different types of features and concatenates them
        num_features = len(self.feature_names)
        assert num_features > 0, "No feature selected: feature_names is empty!"

        assert len(self.feature_names) == len(self.feature_sizes), \
        "length of feature_names (={}) != length of feature_sizes (={})".format( \
        len(self.feature_names), len(self.feature_sizes))

        num_frames = -1  # the number of frames in the video
        feature_matrices = [None
                            ] * num_features  # an array of different features

        for feature_index in range(num_features):
            feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
                features[self.feature_names[feature_index]],
                self.feature_sizes[feature_index], self.max_frames,
                max_quantized_value, min_quantized_value)
            if num_frames == -1:
                num_frames = num_frames_in_this_feature
            #else:
            #  tf.assert_equal(num_frames, num_frames_in_this_feature)

            feature_matrices[feature_index] = feature_matrix

        # cap the number of frames at self.max_frames
        num_frames = tf.minimum(num_frames, self.max_frames)

        # concatenate different features
        video_matrix = feature_matrices[0]
        audio_matrix = feature_matrices[1]

        return contexts["id"], video_matrix, audio_matrix, labels, num_frames


def main(files_pattern):
    data_files = gfile.Glob(files_pattern)
    filename_queue = tf.train.string_input_producer(
        data_files, num_epochs=1, shuffle=False)

    reader = YT8MFrameFeatureReader(
        feature_sizes=[1024, 128], feature_names=["rgb", "audio"])
    vals = reader.prepare_reader(filename_queue)

    with tf.Session() as sess:
        sess.run(tf.initialize_local_variables())
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        vid_num = 0
        all_data = []
        try:
            while not coord.should_stop():
                vid, features, audios, labels, nframes = sess.run(vals)
                label_index = np.where(labels == True)[0].tolist()
                vid_num += 1

                #print vid, features.shape, audios.shape, label_index, nframes

                features_int = features.astype(np.uint8)
                audios_int = audios.astype(np.uint8)

                value_dict = {}
                value_dict['video'] = vid
                value_dict['feature'] = features_int
                value_dict['audio'] = audios_int
                value_dict['label'] = label_index
                value_dict['nframes'] = nframes
                all_data.append(value_dict)

        except tf.errors.OutOfRangeError:
            print('Finished extracting.')

        finally:
            coord.request_stop()
            coord.join(threads)

    print(vid_num)

    record_name = files_pattern.split('/')[-1].split('.')[0]
    outputdir = target_dir
    fn = '%s.pkl' % record_name
    outp = open(os.path.join(outputdir, fn), 'wb')
    pickle.dump(all_data, outp, protocol=2)
    outp.close()


if __name__ == '__main__':
    record_dir = source_dir
    record_files = os.listdir(record_dir)
    for f in record_files:
        record_path = os.path.join(record_dir, f)
        main(record_path)
