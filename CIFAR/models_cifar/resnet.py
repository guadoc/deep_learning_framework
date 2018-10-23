import tensorflow as tf
from layers.activation import relu, lrelu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, svdfc#, residual_block_nb
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv, infodropout

import math
import numpy as np

FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
conv_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_init   = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)

CONV_WEIGHT_DECAY = 0.0005   #0.0001 of 0.0005
FC_WEIGHT_DECAY = 0.0005
regul_conv = weight_decay#shade_conv#
regul_fc = weight_decay#shade#


class Model(Abstract_Model):
    def __init__(self, opts, sess):      
        Abstract_Model.__init__(self, opts, sess)


    def optim_param_schedule(self, board):
        epoch = board.epoch
        momentum = 0.9
        lr = 0.1 * math.pow(0.2, math.floor(monitor.epoch/60))
        # print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
        return {"lr":lr, "momentum":momentum}


    



def wide_block(x, filters_in, filters_out, stride, layer, training_mode):
    conv_params = [ [3,3,stride,stride,1,1], [3,3,1,1,1,1] ]    
    # convs
    with tf.variable_scope('conv_0'):
        xconv, params_bn = bn(x, training_mode)
        xconv = relu(xconv)
        xconv, params = conv_2D(xconv, 3, stride, filters_out, conv_init, use_biases=False, padding='SAME')
        regul_conv(xconv, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=1
          
    with tf.variable_scope('conv_1'):
        xconv, params_bn = bn(xconv, training_mode)
        xconv = relu(xconv)

        x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.9), lambda: tf.nn.dropout(x, 1))
        #x = infodropout(x, "layer_"+str(layer), 1.0, 0.000002, training_mode)   

        xconv, params = conv_2D(xconv, 3, 1, filters_out, conv_init, use_biases=False, padding='SAME')
        regul_conv(xconv, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
        layer+=1
    
    # shortcut
    with tf.variable_scope('shortcut'):
        if filters_in == filters_out:
            xshortcut = x
        else:
            xshortcut, params = conv_2D(x, 1, stride, filters_out, conv_init, use_biases=False, padding='VALID')
            regul_conv(xshortcut, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
            layer+=1
            
    return xconv + xshortcut, layer
    
    def wide_layer(self, x, filters_in, filters_out, count, stride, layer, training_mode):
        with tf.variable_scope('sublayer_0'):
            x, layer = wide_block(x, filters_in, filters_out, stride, layer, training_mode)
        for i in range(1, count):
            with tf.variable_scope('sublayer_'+str(i)):
                x, layer = wide_block(x, filters_out, filters_out, 1, layer, training_mode)
        return x, layer


    def inference(self, inputs, labels, training_mode):    
        x = inputs
        N_LAYER = 0
        N = 4
        k = 10
        nStages = [16, 16*k, 32*k, 64*k]

        N_LAYER+=1
        layer = 1
        
        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, params = conv_2D(x, 3, 1, nStages[0], conv_init, use_biases=True, padding='SAME')
            # regul_conv(x, params, 'layer_'+str(layer), CONV_WEIGHT_DECAY)
            layer+=1
        N_LAYER+=1
        
        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, layer = wide_layer(x, nStages[0], nStages[1], N, 1, layer, training_mode)
        N_LAYER+=1
        
        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, layer = wide_layer(x, nStages[1], nStages[2], N, 2, layer, training_mode)
        N_LAYER+=1
        
        with tf.variable_scope('layer_'+str(N_LAYER)):
            x, layer = wide_layer(x, nStages[2], nStages[3], N, 2, layer, training_mode)
        N_LAYER+=1
        
        x, params_bn = bn(x, training_mode)
        
        x = relu(x)
        x = tf.reduce_mean(relu(x), reduction_indices=[1, 2], name="avg_pool")
        
        with tf.variable_scope('classifier'):
            outputs, params = fc(x, 100, fc_init)
            # regul_fc(outputs, params, 'layer_'+str(layer), FC_WEIGHT_DECAY)
        
        print('WideResNet with '+str(layer) + ' layers')
        
        return outputs, losses, tf.constant(0)