import tensorflow as tf
import numpy as np

def PSNR(target, prediction, name=None):
  with tf.name_scope(name, default_name='psnr_op', values=[target, prediction]):
    squares = tf.square(target-prediction, name='squares')
    squares = tf.reshape(squares, [tf.shape(squares)[0], -1])
    # mean psnr over a batch
    p = tf.reduce_mean((-10/np.log(10))*tf.log(tf.reduce_mean(squares, axis=[1])))
  return p
