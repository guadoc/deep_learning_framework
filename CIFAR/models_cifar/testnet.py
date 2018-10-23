import tensorflow as tf
from layers.activation import relu, bernouilly_activation, input_binary_activation, stochastic_bernouilli_activation
from layers.pooling import max_pool_2D
from layers.trainable import fc, conv_2D
from layers.normalization import bn
from layers.regularization import weight_decay, shade, shade_conv, noize, shade_test
import math

FC_WEIGHT_STDDEV=0.01
CONV_WEIGHT_STDDEV=0.01
conv_init = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
fc_init   = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)



activation = relu
sto_activation = stochastic_bernouilli_activation

# def activation(x):
#     return tf.log(1+tf.nn.relu(x))


def optim_param_schedule(monitor):
    epoch = monitor.epoch
    momentum = 0.9
    lr = 0.07*math.pow(0.98, epoch-1)  #all data
    #lr = 0.007*math.pow(0.98, epoch-1)     
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(CONV_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}

def layer_to_regularize_number():
    return N_LAYERS

N_LAYERS = 20 #12
FC_WEIGHT_DECAY= 0.#001
CONV_WEIGHT_DECAY = 0.#001
regularization_conv = weight_decay#shade_conv
regularization_fc = weight_decay#shade
def layer_regularizer():    
    regs = []
    print('Layer number regularized :'+ str(N_LAYERS))
    for i in range(N_LAYERS):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def add_regul(losses, variables, decay, layer):
    for w in variables:
        tf.add_to_collection("layer_"+str(layer) + "_variables", w)    
    for loss in losses: 
        tf.add_to_collection("layer_"+str(layer) + '_regularization', tf.multiply(loss, decay, name='reg'))    
    return layer + 1



def inception(x, f1_1, f3_3, layer, training_mode, var_list, train_block=True):
    with tf.variable_scope('conv1'):
        x1_1, params_conv1 = conv_2D(x, 1, 1, f1_1, conv_init, use_biases=False, padding='SAME')                        
        regs = regularization_conv(x1_1, params_conv1, 'layer_'+str(layer), CONV_WEIGHT_DECAY)          
        layer = add_regul(regs, params_conv1, layer, CONV_WEIGHT_DECAY)
        x1_1, params_bn1 = bn(x1_1, training_mode)
        if train_block:
            tf.add_to_collection('classification_train', [params_conv1, params_bn1])
    with tf.variable_scope('conv2'):
        x3_3, params_conv2 = conv_2D(x, 3, 1, f3_3, conv_init, use_biases=False, padding='SAME')                        
        regularization_conv(x3_3, params_conv2, 'layer_'+str(layer), CONV_WEIGHT_DECAY)  
        layer = add_regul(regs, params_conv2, layer, CONV_WEIGHT_DECAY)
        x3_3, params_bn2 = bn(x3_3, training_mode)
        if train_block:
            tf.add_to_collection('classification_train', [params_conv2, params_bn2])
    x = tf.concat([x1_1, x3_3], -1)            
    return x, [params_conv1, params_conv2], layer


def downsample(x, filter_conv, layer, training_mode, var_list, train_block=True):
    x_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('conv1'):
        x3_3, params_conv = conv_2D(x, 3, 2, filter_conv, conv_init, use_biases=False, padding='SAME')
        regs = regularization_conv(x3_3, params_conv)
        layer = add_regul(regs, params_conv, layer, CONV_WEIGHT_DECAY)
        x3_3, params_bn = bn(x3_3, training_mode)
    x3_3 = activation(x3_3, training_mode)        
    if train_block:
        tf.add_to_collection('classification_train', [params_conv, params_bn])
    return tf.concat([x_pool, x3_3], -1), [params_conv], layer



def inference(inputs, training_mode):
    x = inputs
    var_list = []
    layer = 1

    #x = input_binary_activation(x, training_mode)  
    # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_mean(x), tf.reduce_max(x)])

    with tf.variable_scope('layer_'+str(layer)):
        n_out = 96
        with tf.variable_scope('conv1'):
            x, params_conv = conv_2D(x, 3, 1, n_out, conv_init, use_biases=False, padding='VALID')
            tf.add_to_collection('classification_train', [params_conv])            
            regs = regularization_conv(x, params_conv)                
            layer+=1
            # x, params_bn = bn(x, training_mode)
            x = activation(x, training_mode)
            # x = bernouilly_activation(x, training_mode)
            

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list = inception(x, 32, 32, layer, training_mode, var_list, train_block=True)
        x = activation(x, training_mode)
    
    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list= inception(x, 32, 48, layer, training_mode, var_list, train_block=True)
        x = activation(x, training_mode)
        

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list = downsample(x, 80, layer, training_mode, var_list, train_block=True)        
        x = activation(x, training_mode)
        

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list = inception(x, 112, 48, layer, training_mode, var_list, train_block=True)        
        x = activation(x, training_mode)

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list = inception(x, 96, 64, layer, training_mode, var_list, train_block=True)        
        x = activation(x, training_mode)
        # x = bernouilly_activation(x, training_mode)

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list = inception(x, 80, 80, layer, training_mode, var_list, train_block=True)        
        x = activation(x, training_mode)

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list = inception(x, 48, 96, layer, training_mode, var_list, train_block=True)        
        x = activation(x, training_mode)

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list = downsample(x, 96, layer, training_mode, var_list, train_block=True)        
        x = activation(x, training_mode)

    with tf.variable_scope('layer_'+str(layer)):
        x, params, layer, var_list= inception(x, 176, 160, layer, training_mode, var_list, train_block=True)        
        x = activation(x, training_mode)
            

    with tf.variable_scope('layer_'+str(layer)):
        #x = bernouilly_activation(x, training_mode)
        x, params, layer, var_list = inception(x, 176, 160, layer, training_mode, var_list, train_block=True)                
        x = activation(x, training_mode)
        # x = bernouilly_activation(x, training_mode)


    x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
    x = tf.reshape(x, [-1, 336])
    

    with tf.variable_scope('layer_'+str(layer)):
        with tf.variable_scope('fc1'):            
            # num_units_in = x.get_shape()[1]
            # num_units_out = 10
            # weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=fc_init, dtype='float')
            # biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer(), dtype='float')
            
            # S,U,V = tf.svd(weights, full_matrices=True)
            # #yact=tf.matmul(x,U)
            # Uk = U[:,0:10]
            # yact = tf.matmul(tf.matmul(x,Uk),tf.transpose(Uk))
            # outputs = tf.nn.xw_plus_b(x, weights, biases)    
            

            outputs, params_fc = fc(x, 10, fc_init)
            reg = regularization_fc(outputs, params_fc, 'layer_'+str(layer), FC_WEIGHT_DECAY)    
            tf.add_to_collection('classification_train', params_fc)


    print("graph for Inception with "+str(layer) + " layers")

    return outputs, [reg[0]]#, yact, x]
