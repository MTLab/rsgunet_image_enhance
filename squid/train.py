#!/usr/bin/env python
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

"""Train a model."""

import argparse
import logging
import numpy as np
import os
import tensorflow as tf
import time


from loss import compute_loss
from net import unet
from data import data_pipeline as dp
from metrics import MultiScaleSSIM
from metrics import PSNR


logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def log_hook(sess, log_fetches):
  """Message display at every log step."""
  data = sess.run(log_fetches)
  step = data['step']
  loss = data['loss']
  loss_content=data['loss_content']
  loss_texture=data['loss_texture']
  loss_color=data['loss_color']
  loss_tv=data['loss_tv']
  loss_Mssim=data['loss_Mssim']
  discim_accuracy = data['discim_accuracy']
  psnr = data['psnr']
  loss_ssim = data['loss_ssim']
  log.info('Step {} | loss = {:.4f}| loss_content = {:.4f} | loss_texture = {:.4f} | '
           'loss_color = {:.4f} | loss_tv = {:.6f} |loss_Mssim = {:.4f} |discim_accuracy =  {:.4f} | psnr = {:.2f} dB|ssim = {:.4f}'.format(step,
                                          loss, loss_content, loss_texture,loss_color,  loss_tv, loss_Mssim,discim_accuracy, psnr, loss_ssim))

def main(args, data_params):
  procname = os.path.basename(args.checkpoint_dir)

  log.info('Preparing summary and checkpoint directory {}'.format(
      args.checkpoint_dir))
  if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

  tf.set_random_seed(1234)  # Make experiments repeatable

  # Select an architecture

  # Add model parameters to the graph (so they are saved to disk at checkpoint)

  # --- Train/Test datasets ---------------------------------------------------
  data_pipe = getattr(dp, args.data_pipeline)
  with tf.variable_scope('train_data'):
    train_data_pipeline = data_pipe(
        args.data_dir,
        shuffle=True,
        batch_size=args.batch_size, nthreads=args.data_threads,
        fliplr=args.fliplr, flipud=args.flipud, rotate=args.rotate,
        random_crop=args.random_crop, params=data_params,
        output_resolution=args.output_resolution,scale=args.scale)
    train_samples = train_data_pipeline.samples


  if args.eval_data_dir is not None:
    with tf.variable_scope('eval_data'):
      eval_data_pipeline = data_pipe(
          args.eval_data_dir,
          shuffle=True,
          batch_size=args.batch_size, nthreads=args.data_threads,
          fliplr=False, flipud=False, rotate=False,
          random_crop=False, params=data_params,
          output_resolution=args.output_resolution,scale=args.scale)
      eval_samples = eval_data_pipeline.samples
  # ---------------------------------------------------------------------------
  swaps = np.reshape(np.random.randint(0, 2, args.batch_size), [args.batch_size, 1])
  swaps = tf.convert_to_tensor(swaps)
  swaps = tf.cast(swaps, tf.float32)
  # Training graph
  with tf.variable_scope('inference'):
    prediction = unet(train_samples['image_input'])
    loss,loss_content,loss_texture,loss_color,loss_Mssim,loss_tv,discim_accuracy =\
      compute_loss.total_loss(train_samples['image_output'], prediction, swaps, args.batch_size)
    psnr = PSNR(train_samples['image_output'], prediction)
    loss_ssim = MultiScaleSSIM(train_samples['image_output'],prediction)


  # Evaluation graph
  if args.eval_data_dir is not None:
    with tf.name_scope('eval'):
      with tf.variable_scope('inference', reuse=True):
        eval_prediction = unet(eval_samples['image_input'])
      eval_psnr = PSNR(eval_samples['image_output'], eval_prediction)
      eval_ssim = MultiScaleSSIM(eval_samples['image_output'], eval_prediction)


  # Optimizer
  model_vars1 = [v for v in tf.global_variables() if v.name.startswith("inference/generator")]
  discriminator_vars1 = [v for v in tf.global_variables() if v.name.startswith("inference/l2_loss/discriminator")]

  global_step = tf.contrib.framework.get_or_create_global_step()
  with tf.name_scope('optimizer'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    updates = tf.group(*update_ops, name='update_ops')
    log.info("Adding {} update ops".format(len(update_ops)))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses and args.weight_decay is not None and args.weight_decay > 0:
      print("Regularization losses:")
      for rl in reg_losses:
        print(" ", rl.name)
      opt_loss = loss + args.weight_decay*sum(reg_losses)
    else:
      print("No regularization.")
      opt_loss = loss

    with tf.control_dependencies([updates]):
      opt = tf.train.AdamOptimizer(args.learning_rate)
      minimize = opt.minimize(opt_loss, name='optimizer', global_step=global_step,var_list=model_vars1)
      minimize_discrim =  opt.minimize(-loss_texture, name='discriminator', global_step=global_step,var_list=discriminator_vars1)


  # Average loss and psnr for display
  with tf.name_scope("moving_averages"):
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_ma = ema.apply([loss,loss_content,loss_texture,loss_color,loss_Mssim,loss_tv,discim_accuracy,psnr,loss_ssim])
    loss = ema.average(loss)
    loss_content=ema.average(loss_content)
    loss_texture=ema.average(loss_texture)
    loss_color=ema.average(loss_color)
    loss_Mssim = ema.average(loss_Mssim)
    loss_tv=ema.average(loss_tv)
    discim_accuracy = ema.average(discim_accuracy)
    psnr = ema.average(psnr)
    loss_ssim = ema.average(loss_ssim)

  # Training stepper operation
  train_op = tf.group(minimize,update_ma)
  train_discrim_op = tf.group(minimize_discrim,update_ma)

  # Save a few graphs to
  summaries = [
    tf.summary.scalar('loss', loss),
    tf.summary.scalar('loss_content',loss_content),
    tf.summary.scalar('loss_color',loss_color),
    tf.summary.scalar('loss_texture',loss_texture),
    tf.summary.scalar('loss_ssim', loss_Mssim),
    tf.summary.scalar('loss_tv', loss_tv),
    tf.summary.scalar('discim_accuracy',discim_accuracy),
    tf.summary.scalar('psnr', psnr),
    tf.summary.scalar('ssim', loss_ssim),
    tf.summary.scalar('learning_rate', args.learning_rate),
    tf.summary.scalar('batch_size', args.batch_size),
  ]

  log_fetches = {
      "loss_content":loss_content,
      "loss_texture":loss_texture,
      "loss_color":loss_color,
      "loss_Mssim": loss_Mssim,
      "loss_tv":loss_tv,
      "discim_accuracy":discim_accuracy,
      "step": global_step,
      "loss": loss,
      "psnr": psnr,
      "loss_ssim":loss_ssim}

  model_vars = [v for v in tf.global_variables() if not v.name.startswith("inference/l2_loss/discriminator")]
  discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("inference/l2_loss/discriminator")]

  # Train config
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU

  sv = tf.train.Supervisor(
      saver=tf.train.Saver(var_list=model_vars, max_to_keep=100),
      local_init_op=tf.initialize_variables(discriminator_vars),
      logdir=args.checkpoint_dir,
      save_summaries_secs=args.summary_interval,
      save_model_secs=args.checkpoint_interval)
  # Train loopl
  with sv.managed_session(config=config) as sess:
    sv.loop(args.log_interval, log_hook, (sess,log_fetches))
    last_eval = time.time()
    while True:
      if sv.should_stop():
        log.info("stopping supervisor")
        break
      try:
        step, _= sess.run([global_step, train_op])
        _ =  sess.run(train_discrim_op)
        since_eval = time.time()-last_eval

        if args.eval_data_dir is not None and since_eval > args.eval_interval:
          log.info("Evaluating on {} images at step {}".format(
              3, step))

          p_ = 0
          s_ = 0
          for it in range(3):
            p_ += sess.run(eval_psnr)
            s_ += sess.run(eval_ssim)
          p_ /= 3
          s_ /= 3

          sv.summary_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="psnr/eval", simple_value=p_)]), global_step=step)

          sv.summary_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="ssim/eval", simple_value=s_)]), global_step=step)

          log.info("  Evaluation PSNR = {:.2f} dB".format(p_))
          log.info("  Evaluation SSIM = {:.4f} ".format(s_))


          last_eval = time.time()

      except tf.errors.AbortedError:
        log.error("Aborted")
        break
      except KeyboardInterrupt:
        break
    chkpt_path = os.path.join(args.checkpoint_dir, 'on_stop.ckpt')
    log.info("Training complete, saving chkpt {}".format(chkpt_path))
    sv.saver.save(sess, chkpt_path)
    sv.request_stop()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # pylint: disable=line-too-long
  # ----------------------------------------------------------------------------
  req_grp = parser.add_argument_group('required')
  req_grp.add_argument('--checkpoint_dir', default='../checkpoint/', help='directory to save checkpoints to.')
  req_grp.add_argument('--data_dir', default= '/root/hj9/ECCV/image_enhance_challenge/train_datasets/dataset.txt', help='input directory containing the training .tfrecords or images.')
  req_grp.add_argument('--eval_data_dir', default= '/root/hj9/ECCV/image_enhance_challenge/test_datasets/dataset.txt', type=str, help='directory with the validation data.')

  # Training, logging and checkpointing parameters
  train_grp = parser.add_argument_group('training')
  train_grp.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for the stochastic gradient update.')
  train_grp.add_argument('--weight_decay', default=None, type=float, help='l2 weight decay on FC and Conv layers.')
  train_grp.add_argument('--log_interval', type=int, default=1, help='interval between log messages (in s).')
  train_grp.add_argument('--summary_interval', type=int, default=120, help='interval between tensorboard summaries (in s)')
  train_grp.add_argument('--checkpoint_interval', type=int, default=600, help='interval between model checkpoints (in s)')
  train_grp.add_argument('--eval_interval', type=int, default=200, help='interval between evaluations (in s)')

  # Debug and perf profiling
  debug_grp = parser.add_argument_group('debug and profiling')
  debug_grp.add_argument('--profiling', dest='profiling', action='store_true', help='outputs a profiling trace.')
  debug_grp.add_argument('--noprofiling', dest='profiling', action='store_false')

  # Data pipeline and data augmentation
  data_grp = parser.add_argument_group('data pipeline')
  data_grp.add_argument('--batch_size', default=32, type=int, help='size of a batch for each gradient update.')
  data_grp.add_argument('--data_threads', default=8, help='number of threads to load and enqueue samples.')
  data_grp.add_argument('--rotate', dest="rotate", action="store_true", help='rotate data augmentation.')
  data_grp.add_argument('--norotate', dest="rotate", action="store_false")
  data_grp.add_argument('--flipud', dest="flipud", action="store_true", help='flip up/down data augmentation.')
  data_grp.add_argument('--noflipud', dest="flipud", action="store_false")
  data_grp.add_argument('--fliplr', dest="fliplr", action="store_true", help='flip left/right data augmentation.')
  data_grp.add_argument('--nofliplr', dest="fliplr", action="store_false")
  data_grp.add_argument('--random_crop', dest="random_crop", action="store_true", help='random crop data augmentation.')
  data_grp.add_argument('--norandom_crop', dest="random_crop", action="store_false")
  data_grp.add_argument('--data_pipeline', default='ImageFilesDataPipeline',help='classname of the data pipeline to use.', choices=dp.__all__)
  data_grp.add_argument('--output_resolution', default=[100, 100], type=int, nargs=2, help='resolution of the output image.')
  data_grp.add_argument('--scale', default= 1, type=int, help='resolution scale of the low image.')

  parser.set_defaults(
      profiling=False,
      flipud=False,
      fliplr=False,
      rotate=False,
      random_crop=True,
      batch_norm=False)
  # ----------------------------------------------------------------------------
  # pylint: enable=line-too-long

  args = parser.parse_args()

  data_params = {}
  for a in data_grp._group_actions:
    data_params[a.dest] = getattr(args, a.dest, None)

  main(args, data_params)




