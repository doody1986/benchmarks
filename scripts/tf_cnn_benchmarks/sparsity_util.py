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

"""Utilities for sparsity analysis."""

import tensorflow as tf

TOWER_NAME = 'tower'

def add_sparsity_summary(x):
  """Helper to create summaries of sparsity.

  Creates a summary that measures the sparsity of a tensor.

  Args:
    x: Tensor
  Returns:
    sparsity: Tensor
  """
  tensor_name = x.op.name
  sparsity = tf.nn.zero_fraction(x)
  tf.summary.scalar(tensor_name + '/sparsity', sparsity)
  return sparsity

def add_sparsity_summary_gradients(loss, x_list):
  """Helper to create summaries for gradients of intermediate results in
  backward pass.

  Creates a summary that measures the sparsity of gradients of intermediate
  results in backward pass.

  Args:
    loss: the loss
    x_list: a list of Tensors
  Returns:
    a list of sparsity of gradients
  """
  gradient_list = tf.gradients(loss, x_list)
  sparsity_list = []
  for g in gradient_list:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = g.op.name
    sparsity = tf.nn.zero_fraction(g)
    tf.summary.scalar(tensor_name + '/sparsity', sparsity)
    sparsity_list.append(sparsity)
  grad_retrieve_list = []
  for i in range(len(gradient_list)):
    grad_retrieve_list.append(gradient_list[i])
    grad_retrieve_list.append(sparsity_list[i])
  return grad_retrieve_list

