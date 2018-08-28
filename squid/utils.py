# Copyright 2016 Google Inc.
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
"""TF graph utilities."""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.stats as st
import cv2

def get_model_params(sess, param_collection="model_params"):
    pcoll = tf.get_collection(param_collection)
    params_ = {p.name.split(':')[0]: p for p in pcoll}
    model_params = sess.run(params_)
    return model_params

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def upsample_layer(x, scale = 2):
  shape = x.get_shape().as_list()
  h = shape[1]
  w = shape[2]
  conv = tf.image.resize_images(x, (scale*h, scale*w))
  conv = slim.conv2d(conv, 3, [1,1], activation_fn = None)
  return conv

def Getfilter(kernel,x):
  return x - blur(kernel,x)


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x,kernel = 21):
    kernel_var = gauss_kernel(kernel, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')



