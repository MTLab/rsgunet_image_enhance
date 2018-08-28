#!/usr/local/bin/bash
# encoding: utf-8
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

"""Evaluates a trained network."""

import argparse
import cv2
import logging
import numpy as np
import os
import re
import setproctitle
import skimage
import skimage.io
import skimage.transform
import sys
sys.path.append("/home/ustc-ee-huangjie/video-practice/hdrnet")
import tensorflow as tf
from ssim import MultiScaleSSIM
import time
import metrics
import models
import utils
import cv2
width = 100
heighth = 100


logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def get_input_list(path):
  regex = re.compile(".*.(png|jpeg|jpg|tif|tiff)")
  if os.path.isdir(path):
    inputs = os.listdir(path)
    inputs = [os.path.join(path, f) for f in inputs if regex.match(f)]
    log.info("Directory input {}, with {} images".format(path, len(inputs)))

  elif os.path.splitext(path)[-1] == ".txt":
    dirname = os.path.dirname(path)
    with open(path, 'r') as fid:
      inputs = [l.strip() for l in fid.readlines()]
    inputs = [os.path.join(dirname, 'input', im) for im in inputs]
    log.info("Filelist input {}, with {} images".format(path, len(inputs)))
  elif regex.match(path):
    inputs = [path]
    log.info("Single input {}".format(path))
  return inputs


def main(args):
  setproctitle.setproctitle('hdrnet_run')

  inputs = get_input_list(args.input)


  # -------- Load params ----------------------------------------------------
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None:
      log.error('Could not find a checkpoint in {}'.format(args.checkpoint_dir))
      return


  # -------- Setup graph ----------------------------------------------------
  tf.reset_default_graph()
  t_fullres_input = tf.placeholder(tf.float32, (1, width, heighth, 3))
  target = tf.placeholder(tf.float32, (1, width, heighth, 3))
  t_lowres_input = utils.blur(5,t_fullres_input)
  img_low = tf.image.resize_images(
                    t_lowres_input, [width/args.scale, heighth/args.scale],
                    method=tf.image.ResizeMethod.BICUBIC)
  img_high = utils.Getfilter(5,t_fullres_input)


  with tf.variable_scope('inference'):
    prediction = models.Resnet(img_low,img_high,t_fullres_input)
  ssim = MultiScaleSSIM(target,prediction)
  psnr = metrics.psnr(target, prediction)
  saver = tf.train.Saver()

  start = time.clock()
  with tf.Session(config=config) as sess:
    log.info('Restoring weights from {}'.format(checkpoint_path))
    saver.restore(sess, checkpoint_path)
    SSIM = 0
    PSNR = 0
    for idx, input_path in enumerate(inputs):
      target_path = args.target + input_path.split('/')[2]
      log.info("Processing {}".format(input_path,target_path))
      im_input = cv2.imread(input_path, -1)  # -1 means read as is, no conversions.
      im_target = cv2.imread(target_path, -1)

      if im_input.shape[2] == 4:
        log.info("Input {} has 4 channels, dropping alpha".format(input_path))
        im_input = im_input[:, :, :3]
        im_target = im_target[:, :, :3]


      im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.
      im_target = np.flip(im_target, 2)


      im_input = skimage.img_as_float(im_input)
      im_target = skimage.img_as_float(im_target)


      im_input = im_input[np.newaxis, :, :, :]
      im_target = im_target[np.newaxis, :, :, :]


      feed_dict = {
          t_fullres_input: im_input,
          target:im_target
      }


      ssim1,psnr1 =  sess.run([ssim,psnr], feed_dict=feed_dict)
      SSIM = SSIM + ssim1
      PSNR = PSNR + psnr1
      if idx>=1000:
        break
    print("SSIM:%s,PSNR:%s"%(SSIM/1000,PSNR/1000))
  end = time.clock()
  print("耗时%s秒"%str(end-start))






if __name__ == '__main__':
  # -----------------------------------------------------------------------------
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_dir', default='checkpoint/', help='path to the saved model variables')
  parser.add_argument('--input', default='test/input/', help='path to the validation data')
  parser.add_argument('--target', default='test/target/', help='path of target data')
  parser.add_argument('--scale', default=2, help ='scale of the image resize')
  # pylint: enable=line-too-long
  # -----------------------------------------------------------------------------

  args = parser.parse_args()
  main(args)
