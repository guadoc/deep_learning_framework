import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.layers.normalization import BatchNormalization


def bn_old(x, is_training, data_format='NHWC'):
    with tf.variable_scope('batch_normalization'):
        momentum = 0.9#0.9997
        epsilon = 0.00001
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]        
        axis = list(range(len(x_shape) - 1))
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
        moving_mean     = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)
        tf.add_to_collection("variable_to_save", moving_mean)
        tf.add_to_collection("variable_to_save", moving_variance)
        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))
        update_moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_mean, mean, momentum), lambda: moving_mean)
        update_moving_variance = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_variance, variance, momentum), lambda: moving_variance)
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon), [beta, gamma]



def bn(x, is_training):
    momemtum = 0.9
    epsilon = 0.00001
    layer_ = BatchNormalization(axis=1,
                                momentum=momemtum,
                                epsilon=epsilon,
                                center=True,
                                scale=True,
                                fused=True)
    x = layer_.apply(x, training=is_training)
    tf.add_to_collection("variable_to_save", layer_.moving_mean)
    tf.add_to_collection("variable_to_save", layer_.moving_variance)    
    # x = tf.Print(x, [ layer_.moving_variance, layer_.moving_mean])
    return x, [layer_.beta, layer_.gamma]

    