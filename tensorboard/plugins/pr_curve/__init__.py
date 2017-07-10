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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import json
import os

import numpy as np
import tensorflow as tf

from tensorboard.plugins.pr_curve import constants

# Contains event-related data specific to the text plugin.
_PrCurvePluginData = namedtuple("_PrCurvePluginData", ["num_thresholds"])

def op(
    tag,
    labels,
    predictions,
    num_thresholds=200,
    weights=None,
    num_classes=1):
  """Computes multi-class PR summaries for a list of thresholds in `[0, 1]`.

  Note: This is basically a faster implementation of simliar methods in
  `tf.contrib.metrics.streaming_XXX_at_thresholds`, where we assume the
  threshold values are evenly distributed and thereby can implement a `O(n+m)`
  algorithm instead of `O(n*m)` in both time and space, where `n` is the
  size of `labels` and `m` is the number of thresholds.

  Computes true/false positive/negative values for the given `predictions`
  against the ground truth `labels`, against a list of evenly distributed
  threshold values in `[0, 1]` of length `num_thresholds`.

  Each number in `predictions`, a float in `[0, 1]`, is compared with its
  corresponding label in `labels`, and counts as a single tp/fp/tn/fn value at
  each threshold. This is then multiplied with `weights` which can be used to
  reweight certain values, or more commonly used for masking values.

  This method supports multi-class classification, and when `num_classes > 1`,
  the last dimension of `labels` and `predictions` is the class dimension.

  Like most `tf.metrics` methods, this returns a pair `(metrics, update_op)`
  where `update_op` should be used at each `session.run()` to update `metrics`,
  and `metrics` should be evaluated at the end.

  Args:
    tag: A tag attached to the summary. Used by TensorBoard for organization.
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Should be of type `bool`.
    predictions: A floating point `Tensor` whose values are in the range
      `[0, 1]`. Shape is arbitrary if `num_classes=1`; otherwise, the
      last dimension should be exactly `num_classes`.
    num_thresholds: Number of thresholds, evenly distributed in `[0, 1]`, to
      compute PR metrics for. Should be `>= 2`.
    weights: Optional; If provided, a `Tensor` that has the same dtype as,
      and broadcastable to, `predictions`.
    num_classes: Optional; Used when `predictions` is a multi-class classifier.

  Returns:
    A tensor summary used by the pr_curve plugin.
  """
  weights = weights if weights is not None else 1.0
  dtype = predictions.dtype

  scope = os.path.join('PRSummaries', tag)
  with tf.name_scope(scope, values=[labels, predictions]):
    # Cast labels to bool then back to float to ensure we have 0.0 or 1.0.
    f_labels = tf.cast(labels, dtype)
    # Ensure predictions are all in range [0.0, 1.0].
    predictions = tf.minimum(1.0, tf.maximum(0.0, predictions))
    # Get weighted true/false labels.
    true_labels = f_labels * weights
    false_labels = (1.0 - f_labels) * weights

    # Before we begin, reshape everything to (num_values, num_classes).
    # We have to do this after weights multiplication since weights broadcast.
    shape = (-1, num_classes)
    predictions = tf.reshape(predictions, shape)
    true_labels = tf.reshape(true_labels, shape)
    false_labels = tf.reshape(false_labels, shape)

    # To compute TP/FP/TN/FN, we are measuring a classifier
    #   C(t) = (predictions >= t)
    # at each threshold 't'. So we have
    #   TP(t) = sum( C(t) * true_labels )
    #   FP(t) = sum( C(t) * false_labels )
    #
    # But, computing C(t) requires computation for each t. To make it fast,
    # observe that C(t) is a cumulative integral, and so if we have
    #   thresholds = [t_0, ..., t_{n-1}];  t_0 < ... < t_{n-1}
    # where n = num_thresholds, and if we can compute the bucket function
    #   B(i) = Sum( (predictions == t), t_i <= t < t{i+1} )
    # then we get
    #   C(t_i) = sum( B(j), j >= i )
    # which is the reversed cumulative sum in tf.cumsum().
    #
    # We can compute B(i) efficiently by taking advantage of the fact that
    # our thresholds are evenly distributed, in that
    #   width = 1.0 / (num_thresholds - 1)
    #   thresholds = [0.0, 1*width, 2*width, 3*width, ..., 1.0]
    # Given a prediction value p, we can map it to its bucket by
    #   bucket_index(p) = floor( p * (num_thresholds - 1) )
    # so we can use tf.scatter_add() to update the buckets in one pass.

    # First compute the bucket indices for each prediction value.
    bucket_indices = (
        tf.cast(tf.floor(predictions * (num_thresholds-1)), tf.int32))
    # Adjust indicies by classes. For performance and simplicity, we keep
    # the buckets (see below) as 1D array representing the tp/fp buckets for
    # a (num_classes, num_thresholds) tensors.
    # So, for each index in bucket_indices, its real index into this 1D array is
    #   index + (num_thresholds * class_index)
    class_indices = tf.constant(range(num_classes), shape=(1, num_classes))
    bucket_indices += num_thresholds * class_indices

    with tf.name_scope('variables'):
      # Now create the variables which correspond to the bucket values.
      # These are flat arrays with num_thresholds values per class.
      tp_buckets_v = tf.contrib.framework.local_variable(
          tf.zeros(
              [num_classes * num_thresholds],
              dtype=dtype),
          name='tp_buckets')
      fp_buckets_v = tf.contrib.framework.local_variable(
          tf.zeros(
              [num_classes * num_thresholds],
              dtype=dtype),
          name='fp_buckets')

    initialize_bucket_counts = tf.variables_initializer(
        [tp_buckets_v, fp_buckets_v])
    with tf.control_dependencies([initialize_bucket_counts]):
      # Create the non-flat views with class_index as first dimension.
      tp_buckets = tf.reshape(tp_buckets_v, (num_classes, -1))
      fp_buckets = tf.reshape(fp_buckets_v, (num_classes, -1))

      with tf.name_scope('update_op'):
        # Use scatter_add to update the buckets.
        update_tp = tf.scatter_add(
            tp_buckets_v, bucket_indices, true_labels, use_locking=True)
        update_fp = tf.scatter_add(
            fp_buckets_v, bucket_indices, false_labels, use_locking=True)

    with tf.control_dependencies([update_tp, update_fp]):
      with tf.name_scope('metrics'):
        thresholds = tf.constant(
            np.linspace(0.0, 1.0, num_thresholds), dtype=dtype)
        # Set up the cumulative sums to compute the actual metrics.
        tp = tf.cumsum(tp_buckets, reverse=True, axis=1, name='tp')
        fp = tf.cumsum(fp_buckets, reverse=True, axis=1, name='fp')
        # fn = sum(true_labels) - tp
        #    = sum(tp_buckets) - tp
        #    = tp[:, 0] - tp
        # Similarly,
        # tn = fp[:, 0] - fp
        tn = tf.subtract(fp[:, 0:1], fp, name='tn')
        fn = tf.subtract(tp[:, 0:1], tp, name='tp')

        # Store the number of thresholds within the summary metadata because
        # that value is constant for all pr curve summaries with the same tag.
        summary_metadata = tf.SummaryMetadata()
        pr_curve_plugin_data = _PrCurvePluginData(num_thresholds=num_thresholds)
        summary_metadata.plugin_data.add(
            plugin_name=constants.PLUGIN_NAME,
            content=json.dumps(pr_curve_plugin_data._asdict()))

        # Store counts within a tensor. We store them in the order:
        # true positives, false positives, true negatives, and false negatives.
        combined_counts = tf.concat([tp, fp, tn, fn], axis=0)
        reshaped_counts = tf.reshape(
            combined_counts, (4, num_classes, num_thresholds))

  return tf.summary.tensor_summary(
      name=tag,
      tensor=tf.cast(reshaped_counts, tf.int64),
      summary_metadata=summary_metadata)

__all__ = ["op"]
