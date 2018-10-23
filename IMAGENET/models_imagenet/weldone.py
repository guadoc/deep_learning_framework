import tensorflow as tf
# from pythoRep.opt_framework.layers.activation import relu, lrelu
# from pythoRep.opt_framework.layers.pooling import max_pool_2D
# from pythoRep.opt_framework.layers.trainable import fc, conv_2D, residual_block
# from pythoRep.opt_framework.layers.normalization import bn
# from pythoRep.opt_framework.layers.regularization import weight_decay, shade, shade_conv
from layers.activation import relu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, residual_block
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np

CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.random_normal_initializer
fc_init = tf.random_normal_initializer(0.01)#tf.uniform_unit_scaling_initializer(factor=1.0)

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    # if epoch < 40:
    #     lr = 0.01
    # elif epoch < 70:
    #     lr = 0.001
    # else:
    #     lr = 0.00005
    if epoch < 2:
        lr = 0.00005
    elif epoch < 5:
        lr = 0.00001
    else:
        lr = 0.000001
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


N_LAYERS_TO_REGULARIZE = 35
regularization_conv = weight_decay
regularization_fc = weight_decay
FC_WEIGHT_DECAY = 0.001
CONV_WEIGHT_DECAY = 0.001
def add_regul(losses, variables, decay, layer):    
    for w in variables:
        tf.add_to_collection("variables_layer_"+str(layer), w)    
    for loss in losses: 
        tf.add_to_collection('regularization_layer_'+str(layer), tf.multiply(loss, decay, name='reg'))    
    return layer + 1


def layers_to_regularize():    
    return range(N_LAYERS_TO_REGULARIZE-1)
    # return []


def block(x, n_block, strides, f_out, activation, training_mode, layer):
    for i in range(n_block):
        # layer += 1
        stride = 1
        if i == 0: stride=strides
        ksizes = [1,3,1]
        strides = [1, stride, 1]
        filters_out = [f_out, f_out, 4*f_out]        
        with tf.variable_scope('res'+str(i+1)):
            x, params_conv, params_bn = residual_block(x, ksizes, strides, filters_out, conv_init, activation, training_mode)            
            regs = regularization_conv(x, params_conv)                
            layer = add_regul(regs, params_conv, CONV_WEIGHT_DECAY, layer)    
            x = activation(x, training_mode)                        
            tf.add_to_collection('classification_train_variables', params_conv)
            tf.add_to_collection('classification_train_variables', params_bn)
    return x, layer


def inference(inputs, training_mode):
    x = inputs
    N_LAYER=1
    layer=0
    with tf.variable_scope('scale_'+str(N_LAYER)):    
        with tf.variable_scope('conv1'):
            n_out = 64            
            x = tf.pad(x, [[0, 0], [0, 0], [3, 3], [3, 3]], "CONSTANT")    
            x, params = conv_2D(x, 7, 2, n_out, conv_init(0.1), False)       
            regs = regularization_conv(x, params)                
            layer = add_regul(regs, params, CONV_WEIGHT_DECAY, layer)        
            x, params_bn = bn(x, training_mode)            
            tf.add_to_collection('classification_train_variables', [params, params_bn])
            x = relu(x, training_mode)                        
            x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT")            
            x = tf.nn.max_pool(x, ksize=[1, 1, 3, 3], strides=[1, 1, 2, 2], padding='VALID', data_format='NCHW')

    N_LAYER=2
    with tf.variable_scope('scale_'+str(N_LAYER)):        
        f_out = 64
        stride = 1        
        x, layer = block(x, 3, stride, f_out, relu, training_mode, layer)        

    N_LAYER=3
    with tf.variable_scope('scale_'+str(N_LAYER)):
        f_out = 128
        stride = 2
        x, layer = block(x, 4, stride, f_out, relu, training_mode, layer)

    N_LAYER=4
    with tf.variable_scope('scale_'+str(N_LAYER)):
        f_out = 256
        stride = 2
        x, layer = block(x, 23, stride, f_out, relu, training_mode, layer)

    N_LAYER=5
    with tf.variable_scope('scale_'+str(N_LAYER)):
        f_out = 512
        stride = 2
        x_, layer = block(x, 3, stride, f_out, relu, training_mode, layer)


    N_LAYER+=1
    with tf.variable_scope('scale_'+str(N_LAYER)):        
        # with tf.variable_scope('conv1'):    
        f_out = 1000
        stride = 1
        x, params = conv_2D(x_, 1, 1, f_out, conv_init, use_biases=True)            
        regs = regularization_conv(x, params)                
        layer = add_regul(regs, params, CONV_WEIGHT_DECAY, layer)    
        tf.add_to_collection('classification_train_variables', params)
            
    x = tf.reshape(x, [- 1, 1000, 14*14])    
    # x = tf.Print(x, [tf.shape(x)])
    # x = tf.transpose(x, [0, 2, 1])
    sorted_val, ind = tf.nn.top_k(  x,    k=196,    sorted=True)    
    outputs = (tf.reduce_sum(sorted_val[:, :, 0:50], axis=2) + tf.reduce_sum(sorted_val[:, :, 146:], axis=2))/50            
    # outputs = tf.Print(outputs, [tf.shape(outputs)])
    
    print('ResNet with '+str(N_LAYER) + ' scales')
    return outputs, [x_, params]
