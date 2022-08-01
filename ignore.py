import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

x = tf.ones((2, 84, 84, 3))
pad = tf.zeros((5, 1, 1, 1))
shape = x.shape
concat = tf.pad(x, paddings=[[0, 0], [0, 0], [0, 0], [0, 5]])
breakpoint()


def generator():
    # The number of observations in the dataset.
    n = 4
    d = 2

    # Integer feature, random from 0 to 4.
    f1 = np.random.randint(0, 5, size=(n, d))

    # String feature.
    strings = np.array([b"cat", b"dog", b"chicken", b"horse", b"goat"])

    features_dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.random.choice([False, True], size=(n, d)),
            # f1,
            # strings[f1],
            np.random.uniform(size=(n, 8, 8)),
        )
    )
    for term, obs in features_dataset:
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        yield tf.train.Example(
            features=tf.train.Features(
                feature={
                    "term": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(term).numpy()]
                        )
                    ),
                    # "f1": tf.train.Feature(int64_list=tf.train.Int64List(value=[f1])),
                    "obs": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(obs).numpy()]
                        )
                    ),
                    # "f3": tf.train.Feature(float_list=tf.train.FloatList(value=[f3])),
                }
            )
        ).SerializeToString()


filename = "test.tfrecord"
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(
    tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
)
example_ds = tf.data.TFRecordDataset(filenames=filename)  # , compression_type="GZIP")


episode_ds = example_ds.map(
    lambda x: tf.io.parse_single_example(
        x,
        {
            "term": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
            "obs": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        },
    )
)


for x in episode_ds:
    breakpoint()
    [term] = x["term"]
    print(tf.io.parse_tensor(term, tf.bool))
    [obs] = x["obs"]
    print(tf.io.parse_tensor(obs, tf.float64))
