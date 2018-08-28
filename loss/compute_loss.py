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
"""Useful image metrics."""

import tensorflow as tf
from squid import utils
import loss.vgg as vgg
from loss.GAN import adversarial
from metrics import MultiScaleSSIM

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
w_content = 0.001
w_color = 0.5
w_texture = 17.5
w_tv = 100
w_ssim = 0.5


# content loss
def content_loss(target, prediction,batch_size):
  CONTENT_LAYER = 'relu5_4'
  CONTENT_LAYER1 = 'relu3_4'
  CONTENT_LAYER2 = 'relu1_2'
  vgg_dir = '/root/hj9/ECCV/image_enhance_challenge/vgg_pretrained/imagenet-vgg-verydeep-19.mat'
  enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(prediction * 255))
  dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(target * 255))

  content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
  content_size1 = utils._tensor_size(dslr_vgg[CONTENT_LAYER1]) * batch_size
  content_size2 = utils._tensor_size(dslr_vgg[CONTENT_LAYER2]) * batch_size

  loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size
  loss_content1 = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER1] - dslr_vgg[CONTENT_LAYER1]) / content_size1
  loss_content2 = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER2] - dslr_vgg[CONTENT_LAYER2]) / content_size2

  return (tf.reduce_mean(loss_content)+tf.reduce_mean(loss_content1)+tf.reduce_mean(loss_content2))/3

#texture loss
def texture_loss(target, prediction, adv_):
  enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(prediction), [-1, PATCH_WIDTH * PATCH_HEIGHT])
  dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(target), [-1, PATCH_WIDTH * PATCH_HEIGHT])
  # enhanced_gray = tf.reshape(prediction, [-1, PATCH_WIDTH * PATCH_HEIGHT*3])
  # dslr_gray = tf.reshape(target, [-1, PATCH_WIDTH * PATCH_HEIGHT*3])
  adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
  adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])
  # adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

  discrim_predictions = adversarial(adversarial_image)

  discrim_target = tf.concat([adv_, 1 - adv_], 1)

  loss_discrim = -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
  loss_texture = -loss_discrim

  correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
  discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  return loss_texture, discim_accuracy



#color loss
def color_loss(target,prediction,batch_size):
    # enhanced_blur = utils.blur(prediction)
    # dslr_blur = utils.blur(target)
    loss_color = tf.reduce_sum(tf.abs(target - prediction)) / (2 * batch_size)
    return loss_color*0.1


def Mssim_loss(target,prediction):
    loss_Mssim = 1-MultiScaleSSIM(target,prediction)
    return loss_Mssim*1000


def tv_loss(prediction,batch_size):
  batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
  tv_y_size = utils._tensor_size(prediction[:, 1:, :, :])
  tv_x_size = utils._tensor_size(prediction[:, :, 1:, :])
  y_tv = tf.nn.l2_loss(prediction[:, 1:, :, :] - prediction[:, :batch_shape[1] - 1, :, :])
  x_tv = tf.nn.l2_loss(prediction[:, :, 1:, :] - prediction[:, :, :batch_shape[2] - 1, :])
  loss_tv = 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size
  return loss_tv


def total_loss(target, prediction, adv_,batch_size,name=None):
  with tf.name_scope(name, default_name='l2_loss', values=[target, prediction]):
    # loss = tf.reduce_mean(tf.square(target-prediction))
    loss_content = content_loss(target,prediction,batch_size)
    loss_texture, discim_accuracy = texture_loss(target,prediction,adv_)
    loss_color = color_loss(target,prediction,batch_size)
    loss_Mssim = Mssim_loss(target, prediction)
    loss_tv = tv_loss(prediction,batch_size)
    loss = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv + w_ssim * loss_Mssim
  return loss, loss_content, loss_texture, loss_color, loss_Mssim, loss_tv, discim_accuracy

