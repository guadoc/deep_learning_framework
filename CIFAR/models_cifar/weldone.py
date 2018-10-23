import tensorflow as tf
from layers.activation import relu, bernouilly_activation
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np

CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.truncated_normal_initializer(stddev=0.01)
fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)



def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    # if epoch < 350:
    #     lr = 0.01
    # else:
    #     lr = 0.001
    # lr = 0.01*math.pow(0.995, monitor.epoch-1) #good one
    lr= 0.015*math.pow(0.98, monitor.epoch-1) #with regul


    # if epoch ==500:
    #     lr =0.
    # elif epoch==530:
    #     lr = 0.001
    # else:
    #     lr=0.0001
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


N_LAYERS_TO_REGULARIZE = 14#29#58#14
CONV_WEIGHT_DECAY = 0.001   #0.0001 of 0.0005
FC_WEIGHT_DECAY = 0.001
regul_conv = weight_decay#shade_conv
regul_fc = weight_decay#shade

def add_regul(losses, variables, decay, layer):    
    for w in variables:
        tf.add_to_collection("variables_layer_"+str(layer), w)    
    for loss in losses: 
        tf.add_to_collection('regularization_layer_'+str(layer), tf.multiply(loss, decay, name='reg'))    
    return layer + 1


def layers_to_regularize():    
    return range(N_LAYERS_TO_REGULARIZE)
    # return [0,1]

activation = relu


def residual_block_nb(x, ksizes, strides, filters_out, initializer, activation, is_training, layer, train_block = True):
    filters_in = x.get_shape()[-1]
    tot_stride = 0
    params_reg = []
    params_train = []
    shortcut = x
    for i in range(len(filters_out)-1):
        with tf.variable_scope('conv'+str(i+1)): 
            x, w = conv_2D(x, ksizes[i], strides[i], filters_out[i], initializer, use_biases=False, padding='SAME')
            tot_stride += strides[i]
            x, params_bn = bn(x, is_training)                    
            x = activation(x, is_training)            
            params_reg+= w
            params_train+=w
            params_train+=params_bn

    with tf.variable_scope('conv'+str(len(filters_out))):             
        x, w = conv_2D(x, ksizes[-1], strides[-1], filters_out[-1], initializer, use_biases=False, padding='SAME')
        tot_stride += strides[-1]
        x, params_bn = bn(x, is_training)                
        params_reg+= w
        params_train+=w
        params_train+=params_bn
        
    if filters_out[-1] != filters_in or tot_stride > len(filters_out):
        with tf.variable_scope('shortcut'):
            shortcut, w = conv_2D(shortcut, 1, tot_stride - len(filters_out)+1, filters_out[-1], initializer, use_biases=False, padding='SAME')            
            params_reg+= w    
            params_train+=w
    with tf.variable_scope('shortcut'):
        shortcut, params_bn = bn(shortcut, is_training)        
        params_train+=params_bn
    out = x + shortcut
    regs = regul_conv(out, params_reg)
    layer = add_regul(regs, params_reg, CONV_WEIGHT_DECAY, layer)
    # layer = add_regul(regs, params_train, CONV_WEIGHT_DECAY, layer)

    if train_block:
        tf.add_to_collection('classification_train', params_train)
    return out, layer



def inference(inputs, training_mode):
    x = inputs
    N_LAYER = 0
    N = 4#9#4
    k = 10#1#10


    layer = 0
    with tf.variable_scope('layer_'+str(N_LAYER)):
        n_out = 16
        x, params = conv_2D(x, 3, 1, n_out, conv_init, use_biases=True, padding='VALID')
        regs = regul_conv(x, [params[0]])
        layer = add_regul(regs, [params[0]], CONV_WEIGHT_DECAY, layer)
        x, params_bn = bn(x, training_mode)
        tf.add_to_collection('classification_train', [params, params_bn])
        # x = bernouilly_activation(x, training_mode)
        x = activation(x, training_mode)


    for i in range(N):
        N_LAYER+=1
        with tf.variable_scope('scale_'+str(N_LAYER)):
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [16*k, 16*k]
            x, layer = residual_block_nb(x, ksizes, strides, filters_out, conv_init, activation, training_mode, layer, train_block = True)
            x = activation(x, training_mode)            


    N_LAYER+=1
    with tf.variable_scope('scale_'+str(N_LAYER)):
        ksizes = [3,3]
        strides = [2, 1]
        filters_out = [32*k, 32*k]
        x, layer = residual_block_nb(x, ksizes, strides, filters_out, conv_init, activation, training_mode, layer, train_block = True)
        x = activation(x, training_mode)
        

    for i in range(N-1):
        N_LAYER+=1
        with tf.variable_scope('scale_'+str(N_LAYER)):
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [32*k, 32*k]
            x, layer = residual_block_nb(x, ksizes, strides, filters_out, conv_init, activation, training_mode, layer, train_block = True)
            # x = bernouilly_activation(x, training_mode)
            x = activation(x, training_mode)            


    N_LAYER+=1
    with tf.variable_scope('scale_'+str(N_LAYER)):
        ksizes = [3,3]
        strides = [2, 1]
        filters_out = [64*k, 64*k]
        x, layer = residual_block_nb(x, ksizes, strides, filters_out, conv_init, activation, training_mode, layer, train_block = True)
        x = activation(x, training_mode)        

    # x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.9), lambda: tf.nn.dropout(x, 1))
    for i in range(N-1):
        N_LAYER+=1
        with tf.variable_scope('scale_'+str(N_LAYER)):
            ksizes = [3,3]
            strides = [1, 1]
            filters_out = [64*k, 64*k]
            x, layer = residual_block_nb(x, ksizes, strides, filters_out, conv_init, activation, training_mode, layer, train_block = True)
            x_ = activation(x, training_mode)
            # x = bernouilly_activation(x, training_mode)
            
    N_LAYER+=1
    with tf.variable_scope('scale_'+str(N_LAYER)):        
        # with tf.variable_scope('conv1'):    
        f_out = 10
        stride = 1
        x, params = conv_2D(x_, 1, 1, f_out, conv_init, use_biases=True)            
        regs = regul_conv(x, params)                
        layer = add_regul(regs, params, CONV_WEIGHT_DECAY, layer)    
        tf.add_to_collection('classification_train_variables', params)
                
    x = tf.reshape(x, [- 1, 10, 7*7])    
    # x = tf.Print(x, [tf.shape(x)])
    # x = tf.transpose(x, [0, 2, 1])
    sorted_val, ind = tf.nn.top_k(  x,    k=49,    sorted=True)    
    # outputs = (tf.reduce_sum(sorted_val[:, :, 0:22], axis=2) + 0.6*tf.reduce_sum(sorted_val[:, :, 38:], axis=2))/33# + 1*tf.reduce_sum(sorted_val[:, :, 33:], axis=2))/49
    
    outputs = (tf.reduce_sum(sorted_val[:, :, 0:37], axis=2) + 0.6*tf.reduce_sum(sorted_val[:, :, 38:], axis=2))/49
    print('ResNet with '+str(layer) + ' layers')    
    return outputs, [x_, params]
