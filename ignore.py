import tensorflow as tf
import numpy as np

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature.
strings = np.array([b"cat", b"dog", b"chicken", b"horse", b"goat"])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution.
feature3 = np.random.randn(n_observations)

features_dataset = tf.data.Dataset.from_tensor_slices(
    (feature0, feature1, feature2, feature3)
)


def generator():
    for feature0, feature1, feature2, feature3 in features_dataset:
        """
        Creates a tf.train.Example message ready to be written to a file.
        """

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = (
                    value.numpy()
                )  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            "feature0": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[feature0])
            ),
            "feature1": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[feature1])
            ),
            "feature2": _bytes_feature(feature2),
            "feature3": tf.train.Feature(
                float_list=tf.train.FloatList(value=[feature3])
            ),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        yield example_proto.SerializeToString()


serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=()
)
filename = "test.tfrecord"
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)