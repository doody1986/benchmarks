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

"""Utilities for monitoring the sparse intermediate results."""

import tensorflow as tf

import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm

def get_non_zero_index(a, shape):
  raw_index = np.where(a != 0)
  n_dim = len(raw_index)
  assert n_dim == 4 or n_dim == 2
  n_data = len(raw_index[0])
  index_list = []
  if n_dim == 4:
    size_chw = shape[1].value * shape[2].value * shape[3].value
    size_hw = shape[2].value * shape[3].value
    size_w = shape[3].value
  elif n_dim == 2:
    size_c = shape[1].value
  for i in range(n_data):
    if n_dim == 4:
      index = raw_index[0][i] * size_chw + raw_index[1][i] * size_hw + raw_index[2][i] * size_w + raw_index[3][i]
    elif n_dim == 2:
      index = raw_index[0][i] * size_c + raw_index[1][i]
    index_list.append(index)
  return index_list

def calc_index_diff_percentage(index_list, ref_index_list, sparsity, all_counts):
  percentage = 1.0
  n_idx = float(len(index_list))
  n_ref_idx = float(len(ref_index_list))
  #print("Current non-zero data size: ", len(index_list))
  #print("Previous non-zero data size: ", len(ref_index_list))
  all_index = np.concatenate((index_list, ref_index_list), axis=0)
  #print("Merged non-zero data size: ", len(all_index))
  #print("Unique non-zero data size: ", len(np.unique(all_index, axis=0)))
  unchanged_counts = len(all_index) - len(np.unique(all_index, axis=0))
  diff_counts = (n_idx - unchanged_counts) + (n_ref_idx - unchanged_counts)
  #print("Differenct counts: ", diff_counts)
  percentage = float(diff_counts) / all_counts
  return percentage

def feature_map_extraction(tensor, batch_index, channel_index):
  # The feature map returned will be represented in a context of matrix
  # sparsity (1 or 0), in which 1 means non-zero value, 0 means zero
  n_dim = len(tensor.shape)
  if n_dim == 4:
    extracted_subarray = tensor[batch_index,:,:,channel_index]
  if n_dim == 2:
    extracted_subarray = tensor
  extracted_subarray[np.nonzero(extracted_subarray)] = 1
  return extracted_subarray


class SparsityMonitor():
  """Logs loss and runtime."""

  def __init__(self, monitor_interval,
             sparsity_threshold, log_animation, batch_idx):
    self._monitor_interval = monitor_interval
    self._sparsity_threshold = sparsity_threshold
    self._log_animation = log_animation
    self._batch_idx = batch_idx

  def before(self, sess):
    self._internal_index_keeper = collections.OrderedDict()
    self._local_step = collections.OrderedDict()
    self._sess = sess

  def after(self, retrieve_list):
    self._data_list = []
    self._sparsity_list = []
    self._eval_results = []
    for tensor in retrieve_list:
      self._eval_results.append(tensor.eval(session=self._sess))

    for i in range(len(self._eval_results)):
      if i % 2 == 0:
        # tensor
        self._data_list.append(self._eval_results[i])
      if i % 2 == 1:
        # sparsity
        self._sparsity_list.append(self._eval_results[i])
    assert len(self._sparsity_list) == len(retrieve_list) / 2
    assert len(self._data_list) == len(retrieve_list) / 2
    num_data = len(self._data_list)
    format_str = ('local_step: %d %s: sparsity = %.2f difference percentage = %.2f')
    for i in range(num_data):
      sparsity = self._sparsity_list[i]
      shape = retrieve_list[2*i].get_shape()
      tensor_name = retrieve_list[2*i].name
      batch_idx = self._batch_idx
      channel_idx = 0
      if tensor_name in self._local_step:
        if self._local_step[tensor_name] == self._monitor_interval and \
           self._log_animation:
          ani = animation.FuncAnimation(fig, animate, frames=self._monitor_interval,
                                        fargs=(tensor_name,),
                                        interval=500, repeat=False, blit=True)                        
          
          figure_name = tensor_name.replace('/', '_').replace(':', '_')
          ani.save(figure_name+'.gif', dpi=80, writer='imagemagick')
          self._local_step[tensor_name] += 1
          continue
        if self._local_step[tensor_name] >= self._monitor_interval:
          continue
      if tensor_name not in self._local_step and sparsity > self._sparsity_threshold:
        self._local_step[tensor_name] = 0
        print (format_str % (self._local_step[tensor_name], tensor_name,
                             sparsity, 0.0))
        self._internal_index_keeper[tensor_name] = get_non_zero_index(self._data_list[i], shape)
        if tensor_name not in data_dict:
          data_dict[tensor_name] = []
        data_dict[tensor_name].append(feature_map_extraction(self._data_list[i], batch_idx, channel_idx))
        self._local_step[tensor_name] += 1
      elif tensor_name in self._local_step and self._local_step[tensor_name] > 0:
        # Inside the monitoring interval
        data_length = self._data_list[i].size
        local_index_list = get_non_zero_index(self._data_list[i], shape)
        diff_percentage = calc_index_diff_percentage(local_index_list,
          self._internal_index_keeper[tensor_name], sparsity, data_length)
        self._internal_index_keeper[tensor_name] = local_index_list
        print (format_str % (self._local_step[tensor_name], tensor_name,
                             sparsity, diff_percentage))
        data_dict[tensor_name].append(feature_map_extraction(self._data_list[i], batch_idx, channel_idx))
        self._local_step[tensor_name] += 1
      else:
        continue
