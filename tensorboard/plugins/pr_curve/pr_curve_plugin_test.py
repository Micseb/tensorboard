# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests the op that generates pr_curve summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import event_multiplexer
from tensorboard.plugins.pr_curve import op


class PrCurveTest(tf.test.TestCase):

  def setUp(self):
    self.logdir = self.get_temp_dir()
    tf.reset_default_graph()

  def test1Class(self):
    # Generate summaries for showing PR curves in TensorBoard.
    sess = tf.Session()
    op(
        tag='tag_bar',
        labels=tf.constant([True, False, True, False], dtype=tf.bool),
        predictions=tf.constant([0.8, 0.6, 0.4, 0.2], dtype=tf.float32),
        num_thresholds=10)
    merged_summary_op = tf.summary.merge_all()
    foo_directory = os.path.join(self.logdir, 'foo')
    writer = tf.summary.FileWriter(foo_directory)
    writer.add_summary(sess.run(merged_summary_op), 1)
    writer.close()

    # Create a multiplexer for reading the data we just wrote.
    multiplexer = event_multiplexer.EventMultiplexer()
    multiplexer.AddRunsFromDirectory(self.logdir)
    multiplexer.Reload()

    # Verify that the metadata was correctly written.
    accumulator = multiplexer.GetAccumulator('foo')
    tag_content_dict = accumulator.PluginTagToContent('pr_curve')
    self.assertListEqual(['tag_bar'], tag_content_dict.keys())
    self.assertDictEqual(
        {'num_thresholds': 10}, json.loads(tag_content_dict['tag_bar']))

    # Test the summary contents.
    tensorEvents = accumulator.Tensors('tag_bar')
    self.assertEqual(1, len(tensorEvents))
    tensorEvent = tensorEvents[0]
    self.assertEqual(1, tensorEvent.step)
    tensorProto = tensorEvent.tensor_proto

    # The tensor shape must be correct. The first dimension is the 4 categories
    # of counts. The 2nd dimension is the number of classes. The last dimension
    # is the number of thresholds.
    self.assertEqual(3, len(tensorProto.tensor_shape.dim))
    correct_shape = [4, 1, 10]
    self.assertListEqual(
        [4, 1, 10], [dim.size for dim in tensorProto.tensor_shape.dim])

    # The values must be correct.
    values = np.reshape(
        np.fromstring(tensorProto.tensor_content, dtype=np.int64),
        correct_shape)
    self.assertListEqual([
        [[2, 2, 2, 2, 1, 1, 1, 1, 0, 0]],
        [[2, 2, 1, 1, 1, 1, 0, 0, 0, 0]],
        [[0, 0, 1, 1, 1, 1, 2, 2, 2, 2]],
        [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2]]
      ], values.tolist())

if __name__ == "__main__":
  tf.test.main()
