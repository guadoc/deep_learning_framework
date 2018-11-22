
import tensorflow as tf
from layers.activation import relu
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D, residual_block
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv
import math
import numpy as np
from abstract_model import Abstract_Model

CONV_WEIGHT_STDDEV = 0.0002
FC_WEIGHT_STDDEV = 0.0002
conv_init = tf.random_normal_initializer
fc_init = tf.random_normal_initializer(0.01)#tf.uniform_unit_scaling_initializer(factor=1.0)


regul_conv = weight_decay
regul_fc = weight_decay
FC_WEIGHT_DECAY = 0.0001
CONV_WEIGHT_DECAY = 0.0001

activation = relu



class Model(Abstract_Model):
    def __init__(self, opts, sess):      
        Abstract_Model.__init__(self, opts, sess)


    def optim_param_schedule(self, board):
        epoch = board.epoch
        momentum = 0.9
        if epoch < 26:
            lr = 0.01
        elif epoch < 49:
            lr = 0.001
        else:
            lr = 0.0001
        # elif epoch < 70:
        #     lr = 0.001
        # else:
        #     lr = 0.00005        
        # print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
        return {"lr":lr, "momentum":momentum}


    def block(self, x, n_block, strides, f_out, activation, training_mode, layer, losses):
        for i in range(n_block):
            # layer += 1
            stride = 1
            if i == 0: stride=strides
            ksizes = [1,3,1]
            strides = [1, stride, 1]
            filters_out = [f_out, f_out, 4*f_out]        
            with tf.variable_scope('res'+str(i+1)):
                x, params_conv, params_bn = residual_block(x, ksizes, strides, filters_out, conv_init, activation, training_mode)
                tf.add_to_collection('classification_loss', params_conv)
                tf.add_to_collection('classification_loss', params_bn)
                layer, losses = self.regularization_conv(layer, losses, CONV_WEIGHT_DECAY, x, params_conv)
                x = activation(x, training_mode)                        
        return x, layer

    def regularization_conv(self, layer, losses, decay, inputs, params):
        reg = regul_conv(inputs, params)
        reg_name = 'regul_'+str(layer)
        losses[reg_name] = reg * decay
        tf.add_to_collection(reg_name, [params])
        return layer + 1, losses


    def inference(self, inputs, labels, training_mode):
        losses = {}
        x = inputs
        N_LAYER=1
        layer=0

        with tf.variable_scope('scale_'+str(N_LAYER)):    
            with tf.variable_scope('conv1'):
                n_out = 64            
                x = tf.pad(x, [[0, 0], [0, 0], [3, 3], [3, 3]], "CONSTANT")    
                x, params_conv = conv_2D(x, 7, 2, n_out, conv_init(0.1), False)       
                x, params_bn = bn(x, training_mode)    
                tf.add_to_collection('classification_loss', params_conv)
                tf.add_to_collection('classification_loss', params_bn)        
                x = activation(x, training_mode)                        
                x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT")            
                x = tf.nn.max_pool(x, ksize=[1, 1, 3, 3], strides=[1, 1, 2, 2], padding='VALID', data_format='NCHW')

        N_LAYER=2
        with tf.variable_scope('scale_'+str(N_LAYER)):        
            f_out = 64
            stride = 1        
            x, layer = self.block(x, 3, stride, f_out, relu, training_mode, layer, losses)        

        N_LAYER=3
        with tf.variable_scope('scale_'+str(N_LAYER)):
            f_out = 128
            stride = 2
            x, layer = self.block(x, 4, stride, f_out, relu, training_mode, layer, losses)

        N_LAYER=4
        with tf.variable_scope('scale_'+str(N_LAYER)):
            f_out = 256
            stride = 2
            x, layer = self.block(x, 23, stride, f_out, relu, training_mode, layer, losses)

        N_LAYER=5
        with tf.variable_scope('scale_'+str(N_LAYER)):
            f_out = 512
            stride = 2
            x, layer = self.block(x, 3, stride, f_out, relu, training_mode, layer, losses)

        
        x = tf.reduce_mean(x, reduction_indices=[2, 3], name="avg_pool")
        N_LAYER=6
        with tf.variable_scope('scale_'+str(N_LAYER)):           
            n_out = 1000
            outputs, params_fc = fc(x, n_out, fc_init)      
            reg = regul_fc(outputs, params_fc)            
            losses['regul_fc'] = reg * FC_WEIGHT_DECAY
            tf.add_to_collection('regul_fc', [params_fc])                  

        tf.add_to_collection('classification_loss', [params_fc])

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels))
        losses['classification_loss'] = cross_entropy

        return outputs, losses, tf.constant(0)
