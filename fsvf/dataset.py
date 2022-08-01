# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLU Atari datasets."""

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.rl_unplugged import atari_utils
from dataclasses import asdict, dataclass

_DESCRIPTION = """
RL Unplugged is suite of benchmarks for offline reinforcement learning. The RL
Unplugged is designed around the following considerations: to facilitate ease of use, we provide the datasets with a unified API which makes it easy for the
practitioner to work with all data in the suite once a general pipeline has been
established.

The datasets follow the [RLDS format](https://github.com/google-research/rlds)
to represent steps and episodes.

"""

EPISODES_PER = 8
DISCOUNT = 0.99
PADDING_VALUE = -1e7
INT_PADDING_VALUE = tf.cast(PADDING_VALUE, tf.int64)


@dataclass
class InitialFeatures:
    actions: Any
    checkpoint_idx: Any
    clipped_rewards: Any
    discounts: Any
    episode_idx: Any
    # # episode_return: Any
    # # clipped_episode_return: Any
    observations: Any
    unclipped_rewards: Any


@dataclass
class Features(InitialFeatures):
    is_first: Any
    is_last: Any
    is_terminal: Any
    return_to_go: Any
    time_step: Any


# Note that rewards and episode_return are actually also clipped.
FEATURE_DESCRIPTION = InitialFeatures(
    actions=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    checkpoint_idx=tf.io.FixedLenFeature([], tf.int64),
    clipped_rewards=tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    discounts=tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    episode_idx=tf.io.FixedLenFeature([], tf.int64),
    observations=tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    unclipped_rewards=tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
)


FEATURES_DICT = tfds.features.FeaturesDict(  # TODO!
    # dict(
    # checkpoint_id=tf.int64,
    # episodes=tfds.features.Dataset(
    #     dict(
    #         steps=tfds.features.Dataset(
    #             dict(
    dict(
        # Features(
        observations=tfds.features.Image(
            shape=(
                84,
                84,
                1,
            ),
            dtype=tf.uint8,
            encoding_format="png",
        ),
        actions=tf.int64,
        return_to_go=tfds.features.Scalar(
            dtype=tf.float32,
            doc=tfds.features.Documentation(desc="Discounted sum of future rewards."),
        ),
        rewards=tfds.features.Scalar(
            dtype=tf.float32,
            doc=tfds.features.Documentation(
                desc="Clipped reward.", value_range="[-1, 1]"
            ),
        ),
        is_terminals=tf.bool,
        is_first=tf.bool,
        is_last=tf.bool,
        # )
    )
    # ),
    #         episode_id=tf.int64,
    #         episode_return=tfds.features.Scalar(
    #             dtype=tf.float32,
    #             doc=tfds.features.Documentation(desc="Sum of the clipped rewards."),
    #          ),
    #     )
    # ),
)


HOMEPAGE = "https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged"


def filename(prefix: str, num_shards: int, shard_id: int):
    return os.fspath(tfds.core.Path(f"{prefix}-{shard_id:05d}-of-{num_shards:05d}"))


def float_tensor_feature(size: int) -> tfds.features.Tensor:
    return tfds.features.Tensor(
        shape=(size,), dtype=tf.float32, encoding=tfds.features.Encoding.ZLIB
    )


def get_files(prefix: str, num_shards: int) -> List[str]:
    return [
        filename(prefix, num_shards, i) for i in range(num_shards)
    ]  # pytype: disable=bad-return-type  # gen-stub-imports


def tf_example_to_step_ds(tf_example: tf.train.Example) -> Dict[str, Any]:
    """Generates an RLDS episode from an Atari TF Example.
    Args:
      tf_example: example from an Atari dataset.
    Returns:
      RLDS episode.
    """

    data = tf.io.parse_single_example(tf_example, asdict(FEATURE_DESCRIPTION))
    episode_length = tf.size(data["actions"])

    is_first = tf.concat(
        [[1], [0] * tf.ones(episode_length - 1, dtype=tf.int64)], axis=0
    )
    is_last = tf.concat(
        [[0] * tf.ones(episode_length - 1, dtype=tf.int64), [1]], axis=0
    )
    is_terminal = [0] * tf.ones_like(data["actions"], dtype=tf.int64)
    time_step = tf.range(episode_length, dtype=tf.int64)

    _discounts = data["discounts"]
    if _discounts[-1] == 0.0:
        is_terminal = tf.concat(
            [[False] * tf.ones(episode_length - 1, tf.int64), [True]], axis=0
        )
        # If the episode ends in a terminal state, in the last step only the
        # observation has valid information (the terminal state).
        _discounts = tf.concat([_discounts[1:], [0.0]], axis=0)

    n1 = tf.cast(tf.math.ceil(episode_length / 2) - 1, tf.int32)
    n2 = episode_length // 2
    p1, p2 = tf.meshgrid(
        tf.range(-n1, episode_length - n1),
        tf.range(-n2, episode_length - n2),
    )
    powers = p1 + p2
    powers = tf.reverse(powers, axis=[0])
    """
    powers:
    [  0  1  2 ... ]
    [ -1  0  1 ... ]
    [ -2 -1  0 ... ]
    ...
    """
    discounts = DISCOUNT ** tf.cast(powers, tf.float32)
    discounts = discounts * tf.cast(powers >= 0, tf.float32)
    """
    dicsounts:
    [  1.00  0.99  0.98 ... ]
    [  0.00  1.00  0.99 ... ]
    [  0.00  0.00  1.00 ... ]
    ...
    """
    rewards = data["unclipped_rewards"]
    rewards = tf.expand_dims(rewards, 0)
    return_to_go = tf.reduce_sum(rewards * discounts, axis=1)

    # noinspection PyUnusedLocal
    def broadcast_idxs(
        checkpoint_idx,
        episode_idx,
        discounts,
        **data,
    ) -> Dict[str, Any]:
        return dict(
            **data,
            checkpoint_idx=[checkpoint_idx] * tf.ones(episode_length, tf.int64),
            discounts=_discounts,
            episode_idx=[episode_idx] * tf.ones(episode_length, tf.int64),
            return_to_go=return_to_go,
        )

    broadcasted = broadcast_idxs(
        **data,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        time_step=time_step,
    )
    return broadcasted


def generate_examples_one_file(
    path: Path,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """Yields examples from one file."""
    # Dataset of tf.Examples containing full episodes.
    example_ds = tf.data.TFRecordDataset(filenames=str(path), compression_type="GZIP")
    # Dataset of episodes, each represented as a dataset of steps.
    episode_ds = example_ds.map(
        tf_example_to_step_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    padding_values = {
        k: tf.cast(PADDING_VALUE, v.dtype)
        for k, v in asdict(FEATURE_DESCRIPTION).items()
        if k != "observations"
    } | dict(
        is_first=INT_PADDING_VALUE,
        is_last=INT_PADDING_VALUE,
        is_terminal=INT_PADDING_VALUE,
        observations=tf.constant("<PAD>"),
        return_to_go=tf.cast(PADDING_VALUE, tf.float32),
        time_step=INT_PADDING_VALUE,
    )

    i = 12
    for sequence in tfds.as_numpy(
        episode_ds.padded_batch(i, padding_values=padding_values).flat_map(
            lambda x: (
                tf.data.Dataset.from_tensor_slices(x)
                .shuffle(int(1e4))
                .batch(
                    10,  # TODO: drop leftovers
                )
            )
        )
    ):
        sequence = Features(**sequence)
        episode_idxs = set(sequence.episode_idx)
        assert max(episode_idxs) - min(episode_idxs) <= 10
        ep_ts = zip(sequence.episode_idx, sequence.time_step)
        record_id = "_".join(f"{ep}.{ts}" for ep, ts in ep_ts)
        assert len(set(sequence.checkpoint_idx))
        yield record_id, asdict(sequence)


class MyRLU(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for RLU Atari."""

    SHARDS = 50
    INPUT_FILE_PREFIX = "gs://rl_unplugged/atari_episodes_ordered"

    VERSION = tfds.core.Version("1.3.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Added is_last.",
        "1.2.0": "Added checkpoint id.",
        "1.3.0": "Removed redundant clipped reward fields.",
    }

    BUILDER_CONFIGS = atari_utils.builder_configs()

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION + atari_utils.description(),
            features=FEATURES_DICT,
            supervised_keys=None,  # disabled
            homepage=HOMEPAGE,
            citation=atari_utils.citation(),
        )

    def get_file_prefix(self):
        run = self.builder_config.run
        game = self.builder_config.game
        return atari_utils.file_prefix(self.INPUT_FILE_PREFIX, run, game)

    def num_shards(self):
        return atari_utils.num_shards(self.builder_config.game, self.SHARDS)

    def get_splits(self):
        paths = {
            "file_paths": get_files(
                prefix=self.get_file_prefix(), num_shards=self.num_shards()
            ),
        }
        return {
            "train": self._generate_examples(paths),
        }

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        del dl_manager

        return self.get_splits()

    @staticmethod
    def _generate_examples(paths):
        """Yields examples."""
        beam = tfds.core.lazy_imports.apache_beam
        file_paths = paths["file_paths"]
        # for p in file_paths:
        # assert "Alien" in inp
        file_paths = file_paths[:1]  # TODO
        return beam.Create(file_paths) | beam.FlatMap(generate_examples_one_file)
