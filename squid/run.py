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
import time
import tensorflow as tf
import models
import utils
import cv2
width = 1536
heighth = 2048


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

    # metapath = ".".join([checkpoint_path, "meta"])
    # log.info('Loading graph from {}'.format(metapath))
    # tf.train.import_meta_graph(metapath)

    # model_params = utils.get_model_params(sess)

  # -------- Setup graph ----------------------------------------------------
  tf.reset_default_graph()
  t_fullres_input = tf.placeholder(tf.float32, (1, width, heighth, 3))
  t_lowres_input = utils.blur(5,t_fullres_input)
  img_low = tf.image.resize_images(
                    t_lowres_input, [width/args.scale, heighth/args.scale],
                    method=tf.image.ResizeMethod.BICUBIC)
  img_high = utils.Getfilter(5,t_fullres_input)


  with tf.variable_scope('inference'):
    prediction = models.Resnet(img_low,img_high,t_fullres_input)
  output = tf.cast(255.0*tf.squeeze(tf.clip_by_value(prediction, 0, 1)), tf.uint8)
  saver = tf.train.Saver()


  with tf.Session(config=config) as sess:
    log.info('Restoring weights from {}'.format(checkpoint_path))
    saver.restore(sess, checkpoint_path)

    for idx, input_path in enumerate(inputs):

      log.info("Processing {}".format(input_path))
      im_input = cv2.imread(input_path, -1)  # -1 means read as is, no conversions.
      if im_input.shape[2] == 4:
        log.info("Input {} has 4 channels, dropping alpha".format(input_path))
        im_input = im_input[:, :, :3]

      im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.

      # log.info("Max level: {}".format(np.amax(im_input[:, :, 0])))
      # log.info("Max level: {}".format(np.amax(im_input[:, :, 1])))
      # log.info("Max level: {}".format(np.amax(im_input[:, :, 2])))

      # HACK for HDR+.
      if im_input.dtype == np.uint16 and args.hdrp:
        log.info("Using HDR+ hack for uint16 input. Assuming input white level is 32767.")
        # im_input = im_input / 32767.0
        # im_input = im_input / 32767.0 /2
        # im_input = im_input / (1.0*2**16)
        im_input = skimage.img_as_float(im_input)
      else:
        im_input = skimage.img_as_float(im_input)

      # Make or Load lowres image
      # lowres_input = skimage.transform.resize(
      #   im_input, [im_input.shape[0]/args.scale, im_input.shape[1]/args.scale], order = 0)
      # im_input = cv2.resize(lowres_input,(2000,1500),interpolation=cv2.INTER_CUBIC)
      # im_input1 = utils.blur(im_input)
      # lowres_input = cv2.resize(im_input1, (im_input1.shape[1]/args.scale,im_input1.shape[0]/args.scale),
      #                           interpolation=cv2.INTER_CUBIC )

      fname = os.path.splitext(os.path.basename(input_path))[0]
      output_path = os.path.join(args.output, fname+".png")
      basedir = os.path.dirname(output_path)

      im_input = im_input[np.newaxis, :, :, :]
      # lowres_input = lowres_input[np.newaxis, :, :, :]

      feed_dict = {
          t_fullres_input: im_input
          # t_lowres_input: lowres_input
      }


      out_ =  sess.run(output, feed_dict=feed_dict)

      if not os.path.exists(basedir):
        os.makedirs(basedir)

      skimage.io.imsave(output_path, out_)




if __name__ == '__main__':
  # ----------------------------------------------------------------------------
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_dir', default='checkpoint/', help='path to the saved model variables')
  parser.add_argument('--input', default='input/', help='path to the validation data')
  parser.add_argument('--output', default='output/', help='path to save the processed images')
  parser.add_argument('--scale', default=2, help ='scale of the image resize')
  # pylint: enable=line-too-long
  # -----------------------------------------------------------------------------

  args = parser.parse_args()
  main(args)
