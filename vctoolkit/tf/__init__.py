import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def tensor_shape(t):
  """
  Get the shape of tensor.

  Parameters
  ----------
  t : tensor
    Input tensor.

  Returns
  -------
  list
    Shape of tensor.
  """
  return t.get_shape().as_list()


def conv(inputs, oc, ks, st, rt=1, pd='SAME'):
  """
  Convolution.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  oc : int
    Number of output channels.
  ks : int
    Kernel size.
  st : int
    Stride.
  rt : int, optional
    Dilation rate, by default 1.
  pd : str
    Padding, by default 'SAME'.

  Returns
  -------
  tensor
    Output tensor.
  """
  layer = tf.layers.conv2d(
    inputs, oc, ks, strides=st, padding=pd, use_bias=False, dilation_rate=rt
  )
  return layer


def conv_bn(inputs, oc, ks, st, scope, training, rt=1, pd='SAME'):
  """
  Convolution - batch normalization.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  oc : int
    Number of output channels.
  ks : int
    Kernel size.
  st : int
    Stride.
  scope : str
    Variable scope.
  training : bool
    Is training or not.
  rt : int, optional
    Dilation rate, by default 1.
  pd : str
    Padding, by default 'SAME'.

  Returns
  -------
  tensor
    Output tensor.
  """
  with tf.variable_scope(scope):
    layer = conv(inputs, oc, ks, st, rt, pd)
    layer = tf.layers.batch_normalization(layer, training=training)
  return layer


def conv_bn_relu(inputs, oc, ks, st, scope, training, rt=1, pd='SAME'):
  """
  Convolution - batch normalization - relu.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  oc : int
    Number of output channels.
  ks : int
    Kernel size.
  st : int
    Stride.
  scope : str
    Variable scope.
  training : bool
    Training or not.
  rt : int, optional
    Dilation rate, by default 1.
  pd : str
    Padding, by default 'SAME'.


  Returns
  -------
  tensor
    Output tensor.
  """
  layer = conv_bn(inputs, oc, ks, st, scope, training, rt, pd)
  layer = tf.nn.relu(layer)
  return layer


def deconv(inputs, oc, ks, st):
  """
  Deconvolution.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  oc : int
    Number of output channels.
  ks : int
    Kernel size.
  st : int
    Stride.

  Returns
  -------
  tensor
    Output tensor.
  """
  layer = tf.layers.conv2d_transpose(
    inputs, oc, ks, st, padding='SAME', use_bias=False
  )
  return layer


def deconv_bn(inputs, oc, ks, st, scope, training):
  """
  Deconvolution - batch normalization.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  oc : int
    Number of output channels.
  ks : int
    Kernel size.
  st : int
    Stride.
  scope : str
    Variable scope.
  training : bool
    Is training or not.

  Returns
  -------
  tensor
    Output tensor.
  """
  with tf.variable_scope(scope):
    layer = deconv(inputs, oc, ks, st)
    layer = tf.layers.batch_normalization(layer, training=training)
  return layer


def deconv_bn_relu(inputs, oc, ks, st, scope, training):
  """
  Deconvolution - batch normalization - relu.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  oc : int
    Number of output channels.
  ks : int
    Kernel size.
  st : int
    Stride.
  scope : str
    Variable scope.
  training : bool
    Training or not.

  Returns
  -------
  tensor
    Output tensor.
  """
  layer = deconv_bn(inputs, oc, ks, st, scope, training)
  layer = tf.nn.relu(layer)
  return layer


def dense(layer, n_units):
  layer = tf.layers.dense(
    layer, n_units, activation=None,
    kernel_initializer=tf.initializers.truncated_normal(stddev=0.01)
  )
  return layer


def dense_bn(layer, n_units, scope, training):
  with tf.variable_scope(scope):
    layer = dense(layer, n_units)
    layer = tf.layers.batch_normalization(layer, training=training)
  return layer


def bottleneck(inputs, oc, st, scope, training, rate=1):
  """
  Bottleneck block for ResNet.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  oc : int
    Number of output channels.
  st : int
    Stride.
  scope : str
    Variable scope.
  training : bool
    Training or not.
  rate : int, optional
    Dilation rate, by default 1

  Returns
  -------
  tensor
    Output tensor.
  """
  with tf.variable_scope(scope):
    ic = inputs.get_shape().as_list()[-1]
    if ic == oc:
      if st == 1:
        shortcut = inputs
      else:
        shortcut = \
          tf.nn.max_pool2d(inputs, [1, st, st, 1], [1, st, st, 1], 'SAME')
    else:
      shortcut = conv_bn(inputs, oc, 1, st, 'shortcut', training)

    residual = conv_bn_relu(inputs, oc//4, 1, 1, 'conv1', training)
    residual = conv_bn_relu(residual, oc//4, 3, st, 'conv2', training, rate)
    residual = conv_bn(residual, oc, 1, 1, 'conv3', training)
    output = tf.nn.relu(shortcut + residual)

  return output


def resnet50(inputs, scope, training, squeeze, size=[3, 4, 6]):
  """
  ResNet 50.

  Parameters
  ----------
  inputs : tensor
    Input tensor.
  scope : str
    Variable scope.
  training : bool
    Training or not.
  squeeze: bool
    Squeeze 1024 channels to 256 channels in the end, or not.
  size: list of 3 numbers
    Number of layers for each block.

  Returns
  -------
  tensor
    Output tensor. If `internal_layer == True`, will also return that layer as
    the second return value.
  """
  with tf.variable_scope(scope):
    layer = conv_bn_relu(inputs, 64, 7, 2, 'conv1', training)

    with tf.variable_scope('block1'):
      for unit in range(size[0] - 1):
        layer = bottleneck(layer, 256, 1, 'unit%d' % (unit + 1), training)
      layer = bottleneck(layer, 256, 2, 'unit3', training)

    with tf.variable_scope('block2'):
      for unit in range(size[1]):
        layer = bottleneck(layer, 512, 1, 'unit%d' % (unit + 1), training, 2)

    with tf.variable_scope('block3'):
      for unit in range(size[2]):
        layer = bottleneck(layer, 1024, 1, 'unit%d' % (unit + 1), training, 4)

    if squeeze:
      layer = conv_bn_relu(layer, 256, 3, 1, 'squeeze', training)

  return layer


def hmap_to_uv(hmap):
  """
  Compute uv coordinates of heat map optima.

  Parameters
  ----------
  hmap : tensor, shape [n, h, w, c]
    Input heat maps.

  Returns
  -------
  tensor, shape [n, c, 2]
    UV coordinates, order (row, column).
  """
  hmap_flat = tf.reshape(hmap, (tf.shape(hmap)[0], -1, tf.shape(hmap)[3]))
  argmax = tf.argmax(hmap_flat, axis=1, output_type=tf.int32)
  argmax_r = argmax // tf.shape(hmap)[2]
  argmax_c = argmax % tf.shape(hmap)[2]
  uv = tf.stack([argmax_r, argmax_c], axis=2)
  return uv
