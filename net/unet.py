import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn



def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def unet(input):
    with tf.variable_scope("generator"):
        input1 = slim.conv2d(input, 16, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv')
        conv1=slim.conv2d(input1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1=slim.conv2d(conv1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        pool1=slim.conv2d(conv1,16,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )

        conv2=slim.conv2d(pool1,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv2=slim.conv2d(conv2,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        pool2=slim.conv2d(conv2,32,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling2' )

        conv3=slim.conv2d(pool2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        conv3=slim.conv2d(conv3,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
        pool3=slim.conv2d(conv3,64,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling3' )


        conv4=slim.conv2d(pool3,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
        conv4=slim.conv2d(conv4,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
        pool4=slim.conv2d(conv4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling4' )


        conv5=slim.conv2d(pool4,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
        conv_global = tf.reduce_mean(conv5,axis=[1,2])
        conv_dense = tf.layers.dense(conv_global,units=128,activation=tf.nn.relu)
        feature = tf.expand_dims(conv_dense,axis=1)
        feature = tf.expand_dims(feature,axis=2)
        ones = tf.zeros(shape=tf.shape(conv4))
        global_feature = feature + ones


        up6 =  tf.concat([conv4, global_feature], axis=3)
        conv6=slim.conv2d(up6,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
        conv6=slim.conv2d(conv6,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, 64, 128  )
        conv7=slim.conv2d(up7,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7=slim.conv2d(conv7,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, 32, 64 )
        conv8=slim.conv2d(up8,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        conv8=slim.conv2d(conv8,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, 16, 32 )
        conv9=slim.conv2d(up9,  16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv9=slim.conv2d(conv9,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

        conv9 = input1 * conv9
        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 16], stddev=0.02))
        conv10 = tf.nn.conv2d_transpose(conv9, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out