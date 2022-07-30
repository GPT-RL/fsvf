import tensorflow as tf
import numpy as np

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def generator():
    # The number of observations in the dataset.
    n = 4

    # Integer feature, random from 0 to 4.
    f1 = np.random.randint(0, 5, n)

    # String feature.
    strings = np.array([b"cat", b"dog", b"chicken", b"horse", b"goat"])

    features_dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.random.choice([False, True], n),
            f1,
            strings[f1],
            np.random.randn(n),
        )
    )
    for f0, f1, f2, f3 in features_dataset:
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        yield tf.train.Example(
            features=tf.train.Features(
                feature={
                    "f0": tf.train.Feature(int64_list=tf.train.Int64List(value=[f0])),
                    "f1": tf.train.Feature(int64_list=tf.train.Int64List(value=[f1])),
                    "f2": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[f2.numpy()])
                    ),
                    "f3": tf.train.Feature(float_list=tf.train.FloatList(value=[f3])),
                }
            )
        ).SerializeToString()


filename = "test.tfrecord"
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(
    tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
)
