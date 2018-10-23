import tensorflow as tf
from layers.activation import relu, bernouilly_activation
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv, reve
import math
from abstract_model import Abstract_Model


FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
conv_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_init   = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)

activation = relu
CONV_WEIGHT_DECAY = 0.0001
regul = shade
def regularization(layer, losses, decay, inputs, params):
    reg = regul(inputs, params)
    reg_name = 'regul_'+str(layer)
    losses[reg_name] = reg * decay
    tf.add_to_collection(reg_name, [params])
    return layer + 1, losses


def regularization_conv(layer, losses, decay, inputs, params):
    reg = shade_conv(inputs, params)
    reg_name = 'regul_'+str(layer)
    losses[reg_name] = reg * decay
    tf.add_to_collection(reg_name, [params])
    return layer + 1, losses

DATA_FORMAT = 'NCHW'

N_REG_LAYER = 5


class Model(Abstract_Model):
    def __init__(self, opts, sess):      
        Abstract_Model.__init__(self, opts, sess)


    def optim_param_schedule(self, board):
        epoch = board.epoch
        momentum = 0.9    
        lr = 0.008*math.pow(0.99, epoch-1)   #good one
        return {"lr":lr, "momentum":momentum}


    def inference(self, inputs, labels, training_mode):
        losses = {}
        
        regul_loss = 0
        x = inputs
        layer = 1
        with tf.variable_scope('layer_1'):
            n_out = 32        
            x, params1 = conv_2D(x, 5, 1, n_out, conv_init, use_biases=True, padding='SAME')
            tf.add_to_collection('classification_loss', [params1])  
            tf.add_to_collection('reve_loss', [params1])                
            layer, losses = regularization_conv(layer, losses, CONV_WEIGHT_DECAY, x, params1)            
            x = activation(x, training_mode)
            x = tf.nn.max_pool(x, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding='SAME', data_format=DATA_FORMAT)
        
        with tf.variable_scope('layer_2'):
            n_out = 64
            x, params2 = conv_2D(x, 5, 1, n_out, conv_init, True, padding='SAME')                 
            tf.add_to_collection('classification_loss', [params2])   
            tf.add_to_collection('reve_loss', [params2])   
            layer, losses = regularization_conv(layer, losses, CONV_WEIGHT_DECAY, x, params2)            
            x = activation(x, training_mode)
            x = tf.nn.max_pool(x, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding='SAME', data_format=DATA_FORMAT)

        
        with tf.variable_scope('layer_3'):
            n_out = 64
            x, params3 = conv_2D(x, 5, 1, n_out, conv_init, True, padding='SAME')            
            tf.add_to_collection('classification_loss', [params3])
            tf.add_to_collection('reve_loss', [params3])
            layer, losses = regularization_conv(layer, losses, CONV_WEIGHT_DECAY, x, params3)            
            x = activation(x, training_mode)
            x = tf.nn.max_pool(x, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding='SAME', data_format=DATA_FORMAT)

        x_ = tf.reshape(x, [-1, 4*4*n_out])        
        with tf.variable_scope('layer_4'):
            n_out = 1000
            x, params4 = fc(x_, n_out, fc_init)
            tf.add_to_collection('classification_loss', [params4])
            tf.add_to_collection('reve_loss', [params4])
            layer, losses = regularization(layer, losses, CONV_WEIGHT_DECAY, x, params4)
            x = activation(x, training_mode)

        with tf.variable_scope('layer_5'):
            n_outputs = 10
            outputs, params5 = fc(x, n_outputs, fc_init)  
            tf.add_to_collection('classification_loss', [params5])
            tf.add_to_collection('reve_loss', [params5])
            layer, losses = regularization(layer, losses, CONV_WEIGHT_DECAY, outputs, params5)             

            reve_loss  = reve(x, params5, 0.002, labels)

            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels))        
            losses['classification_loss'] = cross_entropy

        return outputs, losses, tf.constant(0)