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

cmap = ListedColormap(['black', 'red'])
fig, ax = plt.subplots()

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

def feature_map_extraction(tensor, data_format, batch_index, channel_index):
  # The feature map returned will be represented in a context of matrix
  # sparsity (1 or 0), in which 1 means non-zero value, 0 means zero
  n_dim = len(tensor.shape)
  if n_dim == 4:
    if data_format == "NCHW":
      extracted_subarray = tensor[batch_index,channel_index,:,:]
    elif data_format == "NHWC":
      extracted_subarray = tensor[batch_index,:,:,channel_index]
  if n_dim == 2:
    extracted_subarray = tensor
  extracted_subarray[np.nonzero(extracted_subarray)] = 1
  return extracted_subarray

def animate(i, ax, tensor_name, data_dict):
  label = 'Local step in monitoring period: {0}'.format(i)
  matrix = data_dict[tensor_name][i] 
  mesh = ax.pcolormesh(matrix, cmap=cmap)
  ax.set_xlabel(label)
  return mesh,


class SparsityMonitor():
  """Logs loss and runtime."""

  def __init__(self, data_format, monitor_interval,
             sparsity_threshold, log_animation, batch_idx):
    self._data_format = data_format
    self._monitor_interval = monitor_interval
    self._sparsity_threshold_list = np.arange(sparsity_threshold, 1, 0.1)
    self._initial_sparsity_threshold = self._sparsity_threshold_list[0]
    self._log_animation = log_animation
    self._batch_idx = batch_idx
    self._extracted_data_dict = collections.OrderedDict()
    self._internal_index_keeper = collections.OrderedDict()
    self._local_step = collections.OrderedDict()

    self._current_threshold_idx = collections.OrderedDict()
    self._sparsity_threshold = collections.OrderedDict()
    self._fd_dict = collections.OrderedDict()
    self._finished = collections.OrderedDict()

  def _reset(self, tensor_name):
    if tensor_name in self._extracted_data_dict:
      self._extracted_data_dict[tensor_name] = []
    if tensor_name in self._internal_index_keeper:
      self._internal_index_keeper[tensor_name] = []
    if tensor_name in self._local_step:
      self._local_step[tensor_name] = 0
    self._current_threshold_idx[tensor_name] += 1
    self._sparsity_threshold[tensor_name] = \
      self._sparsity_threshold_list[self._current_threshold_idx[tensor_name]]
    self._fd_dict[tensor_name] = None

  def collect(self, results, retrieve_list):
    self._data_list = []
    self._sparsity_list = []

    for i in range(len(retrieve_list)):
      if i % 2 == 0:
        # tensor
        self._data_list.append(results[retrieve_list[i].name])
      if i % 2 == 1:
        # sparsity
        self._sparsity_list.append(results[retrieve_list[i].name])
    assert len(self._sparsity_list) == len(retrieve_list) / 2
    assert len(self._data_list) == len(retrieve_list) / 2
    num_data = len(self._data_list)
    format_str = ('local_step: %d %s: sparsity = %.2f difference percentage = %.2f\n')
    for i in range(num_data):
      sparsity = self._sparsity_list[i]
      shape = retrieve_list[2*i].get_shape()
      tensor_name = retrieve_list[2*i].name
      batch_idx = self._batch_idx
      channel_idx = 0
      if tensor_name in self._finished and self._finished[tensor_name]:
        continue
      if sparsity < self._initial_sparsity_threshold:
        continue

      if tensor_name in self._local_step:
        if self._local_step[tensor_name] == self._monitor_interval and \
           self._log_animation:
          fig, ax = plt.subplots()
          ani = animation.FuncAnimation(fig, animate, frames=self._monitor_interval,
                                        fargs=(ax, tensor_name, self._extracted_data_dict,),
                                        interval=500, repeat=False, blit=True)                        
          
          figure_name = tensor_name.replace('/', '_').replace(':', '_') +\
                        str(int(self._sparsity_threshold[tensor_name] * 100))
          ani.save(figure_name+'.gif', dpi=80, writer='imagemagick')
          self._local_step[tensor_name] += 1
          continue
        if self._local_step[tensor_name] >= self._monitor_interval:
          self._fd_dict[tensor_name].close()
          if self._current_threshold_idx[tensor_name] < (len(self._sparsity_threshold_list)-1):
            self._reset(tensor_name)
          elif self._current_threshold_idx[tensor_name] == (len(self._sparsity_threshold_list)-1):
            self._finished[tensor_name] = True
          continue
      if tensor_name not in self._local_step and sparsity > self._initial_sparsity_threshold:
        # Initial setup for each structure
        self._local_step[tensor_name] = 0
        self._current_threshold_idx[tensor_name] = 0
        self._sparsity_threshold[tensor_name] = self._sparsity_threshold_list[self._current_threshold_idx[tensor_name]]
        self._finished[tensor_name] = False
        self._internal_index_keeper[tensor_name] = get_non_zero_index(self._data_list[i], shape)
        if tensor_name not in self._extracted_data_dict:
          self._extracted_data_dict[tensor_name] = []
        self._extracted_data_dict[tensor_name].append(feature_map_extraction(\
                                            self._data_list[i],\
                                            self._data_format,\
                                            batch_idx, channel_idx))
        self._local_step[tensor_name] += 1
        if tensor_name not in self._fd_dict:
          file_name = tensor_name.replace('/', '_').replace(':', '_') +\
                      str(int(self._initial_sparsity_threshold * 100)) + '.txt'
          self._fd_dict[tensor_name] = open(file_name, 'w')
        self._fd_dict[tensor_name].write(
                      format_str % (self._local_step[tensor_name], tensor_name,
                      sparsity, 0.0))
      elif tensor_name in self._local_step and self._local_step[tensor_name] == 0\
           and sparsity > self._sparsity_threshold[tensor_name]:
        # After reset
        self._internal_index_keeper[tensor_name] = get_non_zero_index(self._data_list[i], shape)
        assert(self._extracted_data_dict[tensor_name] == [])
        self._extracted_data_dict[tensor_name].append(feature_map_extraction(\
                                            self._data_list[i],\
                                            self._data_format,\
                                            batch_idx, channel_idx))
        self._local_step[tensor_name] += 1
        file_name = tensor_name.replace('/', '_').replace(':', '_') +\
                    str(int(self._sparsity_threshold[tensor_name] * 100)) + '.txt'
        assert(self._fd_dict[tensor_name] == None)
        self._fd_dict[tensor_name] = open(file_name, 'w')
        self._fd_dict[tensor_name].write(
                      format_str % (self._local_step[tensor_name], tensor_name,
                      sparsity, 0.0))
      elif tensor_name in self._local_step and self._local_step[tensor_name] > 0:
        # Inside the monitoring interval
        data_length = self._data_list[i].size
        local_index_list = get_non_zero_index(self._data_list[i], shape)
        diff_percentage = calc_index_diff_percentage(local_index_list,
          self._internal_index_keeper[tensor_name], sparsity, data_length)
        self._internal_index_keeper[tensor_name] = local_index_list
        self._extracted_data_dict[tensor_name].append(feature_map_extraction(
                                            self._data_list[i],\
                                            self._data_format,\
                                            batch_idx, channel_idx))
        self._local_step[tensor_name] += 1
        self._fd_dict[tensor_name].write(
                      format_str % (self._local_step[tensor_name], tensor_name,
                      sparsity, diff_percentage))
      else:
        continue

