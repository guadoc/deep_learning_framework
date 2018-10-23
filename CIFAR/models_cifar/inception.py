import tensorflow as tf
from layers.activation import relu, bernouilly_activation, input_binary_activation, stochastic_bernouilli_activation
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv, noize
import math


N_CLASSES = 10

FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
conv_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_init   = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)

activation = relu
DATA_FORMAT='NCHW'#'NHWC'

def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.07*math.pow(0.98, epoch-1)  #all data
    #lr = 0.007*math.pow(0.98, epoch-1)     
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}


FC_WEIGHT_DECAY= 0.0001
CONV_WEIGHT_DECAY = 0.0001
regularization_conv = shade_conv
regularization_fc = shade#weight_decay


def add_loss(losses, loss, variables, name, decay=1):
    for w in variables:
        tf.add_to_collection(name, w)    
    losses[name] = loss*decay
    return losses


def inception(x, f1_1, f3_3, layer, losses, training_mode, train_block=True):    
    layer+=1
    with tf.variable_scope('layer_'+str(layer)):
        with tf.variable_scope('conv1'):
            x1_1, params_conv1 = conv_2D(x, 1, 1, f1_1, conv_init, use_biases=False, padding='SAME')            
        with tf.variable_scope('conv2'):
            x3_3, params_conv2 = conv_2D(x, 3, 1, f3_3, conv_init, use_biases=False, padding='SAME')
        x = tf.concat([x1_1, x3_3], -3)     
        reg_loss = regularization_conv(x3_3, [params_conv1[0], params_conv2[0]])    
        x, params_bn = bn(x, training_mode)   
    if train_block:
        tf.add_to_collection('classification_train_variables', [params_conv1, params_conv2, params_bn])
    losses = add_loss(losses, reg_loss, [params_conv1[0], params_conv2[0]], 'reg_layer_'+str(layer), decay=CONV_WEIGHT_DECAY)
    return x, layer, losses


def downsample(x, filter_conv, layer, losses, training_mode, train_block=True):
    layer+=1
    with tf.variable_scope('layer_'+str(layer)):
        x_pool = tf.nn.max_pool(x, ksize=[1, 1, 3, 3], strides=[1, 1, 2, 2], padding='SAME', data_format=DATA_FORMAT)
        with tf.variable_scope('conv1'):
            x3_3, params_conv = conv_2D(x, 3, 2, filter_conv, conv_init, use_biases=False, padding='SAME')    
        x = tf.concat([x_pool, x3_3], -3)
        reg_loss = regularization_conv(x, params_conv)    
        x, params_bn = bn(x, training_mode)
        x = activation(x, training_mode)        
        if train_block:
            tf.add_to_collection('classification_train_variables', [params_conv, params_bn])
    losses = add_loss(losses, reg_loss, params_conv, 'reg_layer_'+str(layer), decay=CONV_WEIGHT_DECAY)
    return x, layer, losses


def classifier(x):
    outputs, params = fc(x, N_CLASSES, fc_init)
    tf.add_to_collection('classification_train_variables', params)
    return outputs, params 



def inference(inputs, labels, training_mode):
    x = inputs
    layer = 0
    losses = {}

    layer +=1
    with tf.variable_scope('layer_'+str(layer)):
        n_out = 96
        with tf.variable_scope('conv1'):
            x, params_conv = conv_2D(x, 2, 1, n_out, conv_init, use_biases=False, padding='VALID')            
            reg_loss = regularization_conv(x, params_conv)                            
            x, params_bn = bn(x, training_mode)
            tf.add_to_collection('classification_train_variables', [params_conv, params_bn])            
            x = activation(x, training_mode)# x = bernouilly_activation(x, training_mode)
            losses = add_loss(losses, reg_loss, params_conv, 'reg_layer_'+str(layer), decay=CONV_WEIGHT_DECAY)
    
    x, layer, losses = inception(x, 32, 32, layer, losses, training_mode, train_block=True)
    x = activation(x, training_mode)
    
    x, layer, losses = inception(x, 32, 48, layer, losses, training_mode, train_block=True)
    x = activation(x, training_mode)
        

    x, layer, losses = downsample(x, 80, layer, losses, training_mode, train_block=True)        
    x = activation(x, training_mode)
        


    x, layer, losses = inception(x, 112, 48, layer, losses, training_mode, train_block=True)        
    x = activation(x, training_mode)


    x, layer, losses = inception(x, 96, 64, layer, losses, training_mode, train_block=True)        
    x = activation(x, training_mode)

    
    x, layer, losses = inception(x, 80, 80, layer, losses, training_mode, train_block=True)        
    x = activation(x, training_mode)

    x, layer, losses = inception(x, 48, 96, layer, losses, training_mode, train_block=True)        
    x = activation(x, training_mode)


    x, layer, losses = downsample(x, 96, layer, losses, training_mode, train_block=True)        
    x = activation(x, training_mode)


    x, layer, losses = inception(x, 176, 160, layer, losses, training_mode, train_block=True)        
    x = activation(x, training_mode)
            

    x, layer, losses = inception(x, 176, 160, layer, losses, training_mode, train_block=True)                
    x = activation(x, training_mode)


    x = tf.nn.avg_pool(x, ksize=[1, 1, 7, 7], strides=[1, 1, 1, 1], padding='VALID', data_format=DATA_FORMAT)
    x = tf.reshape(x, [-1, 336])
    

    # with tf.variable_scope('layer_'+str(layer)):
    #     with tf.variable_scope('fc1'):   

    #         # num_units_in = x.get_shape()[1]
    #         # weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=fc_init, dtype='float')
    #         # biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer(), dtype='float')            
    #         # S,U,V = tf.svd(weights, full_matrices=True)            
    #         # Uk = U[:,0:10]
    #         # yact = tf.matmul(tf.matmul(x,Uk),tf.transpose(Uk))
    #         # outputs = tf.nn.xw_plus_b(x, weights, biases)    

    #         outputs, params_fc = fc(x, num_units_out, fc_init)
    #         # regs = regularization_fc(outputs, params_fc)
    #         regs = regularization_fc(x, params_fc)
    #         layer = add_regul(regs, params_conv, CONV_WEIGHT_DECAY, layer)
    #         tf.add_to_collection('classification_train', params_fc)


    
    ########### For vshade
    with tf.variable_scope('classifier'):
        outputs, params_fc = classifier(x)    

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels))    
    losses = add_loss(losses, cross_entropy, tf.get_collection('classification_train_variables'), 'classif_loss', decay=1)    

    print("graph for Inception with "+str(layer) + " layers")
    return outputs, losses, [x, params_fc]
