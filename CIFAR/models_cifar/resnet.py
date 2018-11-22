import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv, reve
from abstract_model import Abstract_Model

import math
import numpy as np

FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
conv_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_init   = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)

CONV_WEIGHT_DECAY = 0.0005   #0.0001 
FC_WEIGHT_DECAY = 0.0005
regul_conv = weight_decay#shade_conv
regul_fc = weight_decay#shade#

activation = relu


def regularization_fc(layer, losses, decay, inputs, params):
    reg = regul_fc(inputs, params)
    reg_name = 'regul_'+str(layer)
    losses[reg_name] = reg * decay
    tf.add_to_collection(reg_name, [params])
    return layer + 1, losses


def regularization_conv(layer, losses, decay, inputs, params):
    reg = regul_conv(inputs, params)
    reg_name = 'regul_'+str(layer)
    losses[reg_name] = reg * decay
    tf.add_to_collection(reg_name, [params])
    return layer + 1, losses


class Model(Abstract_Model):
    def __init__(self, opts, sess):      
        Abstract_Model.__init__(self, opts, sess)


    def optim_param_schedule(self, board):
        epoch = board.epoch
        momentum = 0.9
        lr = 0.1 * math.pow(0.2, math.floor(epoch/60))        
        return {"lr":lr, "momentum":momentum}

    
    def wide_block(self, x, filters_in, filters_out, stride, layer, training_mode, losses):
        with tf.variable_scope('conv_0'):
            xconv, params_bn = bn(x, training_mode)
            xconv = activation(xconv, training_mode)
            xconv, params_conv = conv_2D(xconv, 3, stride, filters_out, conv_init, use_biases=False, padding='SAME')
            tf.add_to_collection('classification_loss', [params_conv])
            tf.add_to_collection('classification_loss', [params_bn])
            tf.add_to_collection('reve_loss', [params_conv])
            tf.add_to_collection('reve_loss', [params_bn])
            layer, losses  = regularization_conv(layer, losses, CONV_WEIGHT_DECAY, xconv, params_conv)

        with tf.variable_scope('conv_1'):
            xconv, params_bn = bn(xconv, training_mode)
            xconv = activation(xconv, training_mode)                
            xconv, params_conv = conv_2D(xconv, 3, 1, filters_out, conv_init, use_biases=False, padding='SAME')
            tf.add_to_collection('classification_loss', [params_conv])
            tf.add_to_collection('classification_loss', [params_bn])
            tf.add_to_collection('reve_loss', [params_conv])
            tf.add_to_collection('reve_loss', [params_bn])
            layer, losses  = regularization_conv(layer, losses, CONV_WEIGHT_DECAY, xconv, params_conv)
        # shortcut
        with tf.variable_scope('shortcut'):
            if filters_in == filters_out:
                xshortcut = x
            else:
                xshortcut, params_conv = conv_2D(x, 1, stride, filters_out, conv_init, use_biases=False, padding='VALID')
                layer, losses  = regularization_conv(layer, losses, CONV_WEIGHT_DECAY, xshortcut, params_conv)
                tf.add_to_collection('classification_loss', [params_conv])                                
                tf.add_to_collection('reve_loss', [params_conv])
        return xconv + xshortcut, layer, losses
    

    def wide_layer(self, x, filters_in, filters_out, count, stride, layer, training_mode, losses):
        with tf.variable_scope('sublayer_0'):
            x, layer, losses = self.wide_block(x, filters_in, filters_out, stride, layer, training_mode, losses)
        for i in range(1, count):
            with tf.variable_scope('sublayer_'+str(i)):
                x, layer, losses = self.wide_block(x, filters_out, filters_out, 1, layer, training_mode, losses)
        return x, layer, losses


    def inference(self, inputs, labels, training_mode):    
        x = inputs
        N_LAYER = 0
        N = 4
        k = 10
        nStages = [16, 16*k, 32*k, 64*k]
        losses = {}

        N_LAYER+=1
        layer = 1
        
        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, params_conv = conv_2D(x, 3, 1, nStages[0], conv_init, use_biases=True, padding='SAME')
            layer, losses  = regularization_conv(layer, losses, CONV_WEIGHT_DECAY, x, params_conv)
            tf.add_to_collection('classification_loss', [params_conv])
            tf.add_to_collection('reve_loss', [params_conv])
        N_LAYER+=1        

        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, layer, losses= self.wide_layer(x, nStages[0], nStages[1], N, 1, layer, training_mode, losses)
        N_LAYER+=1
        
        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, layer , losses= self.wide_layer(x, nStages[1], nStages[2], N, 2, layer, training_mode, losses)
        N_LAYER+=1
        
        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, layer, losses = self.wide_layer(x, nStages[2], nStages[3], N, 2, layer, training_mode, losses)
        N_LAYER+=1
        
        x, params_bn = bn(x, training_mode)
        tf.add_to_collection('classification_loss', [params_bn])        
        tf.add_to_collection('reve_loss', [params_bn])
        
        x = activation(x, training_mode)
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
        
        with tf.variable_scope('classifier'):
            outputs, params_fc = fc(x, 100, fc_init)
            tf.add_to_collection('classification_loss', [params_fc])
            tf.add_to_collection('reve_loss', [params_fc])
            layer, losses  = regularization_fc(layer, losses, FC_WEIGHT_DECAY, outputs, params_conv)
        
        losses['reve_loss'] = reve(x, params_fc, 0.002, labels)
        losses['classification_loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)) 
        
        print('WideResNet with '+str(layer) + ' layers')
        
        return outputs, losses, tf.constant(0)