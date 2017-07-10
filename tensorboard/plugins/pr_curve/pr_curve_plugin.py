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
"""The TensorBoard PR curve plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imghdr

from six.moves import urllib
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins import base_plugin

_PLUGIN_PREFIX_ROUTE = 'pr_curve'


class PrCurvePlugin(base_plugin.TBPlugin):
  """PR Curve Plugin for TensorBoard."""

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def __init__(self, context):
    """Instantiates PrCurvePlugin via TensorBoard core.

    Args:
      context: A base_plugin.TBContext instance.
    """
    self._multiplexer = context.multiplexer
