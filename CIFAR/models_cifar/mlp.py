import tensorflow as tf
import math
from layers.trainable import fc
from layers.activation import relu as activation, bernouilly_activation
from layers.regularization import weight_decay, shade 




FC_WEIGHT_STDDEV=0.01
fc_init = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)



def optim_param_schedule(monitor):#CIFAR10
    epoch = monitor.epoch
    momentum = 0.9
    # lr = 0.01*math.pow(0.97, epoch-1) #good one
    if epoch ==500:
        lr =0.
    else:
        lr = 0.001
        #lr = 0.001*math.pow(0.97, epoch-1) #good one
    print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(FC_WEIGHT_DECAY))
    return {"lr":lr, "momentum":momentum}

# def optim_param_schedule(monitor):#CIFAR100
#     epoch = monitor.epoch
#     momentum = 0.9
#     lr = 0.04*math.pow(0.96, epoch-1)
#     print("lr: "+str(lr)+ ", momentum: "+str(momentum) + ", decay: "+str(FC_WEIGHT_DECAY))
#     return {"lr":lr, "momentum":momentum}



regularization = weight_decay#shade
FC_WEIGHT_DECAY= 0.#0005
N_LAYERS = 4
def layer_regularizer():    
    regs = []
    print('Layer number regularized :'+ str(N_LAYERS))
    for i in range(N_LAYERS):
        regs.append([tf.reduce_sum(tf.get_collection('layer_'+str(i+1)+'_reg')), tf.get_collection('layer_'+str(i+1)+'_variables')])
    return regs

def inference(inputs, training_mode):
    x = tf.reshape(inputs, [-1, 28*28*3])

    with tf.variable_scope('layer_4'):
        n_out = 512
        x, params = fc(x, n_out, fc_init)
        # tf.add_to_collection('classification_train', params)
        regularization(x, params, tf.get_variable_scope().name, FC_WEIGHT_DECAY)
        x = bernouilly_activation(x, training_mode)
        x = activation(x, training_mode)

    with tf.variable_scope('layer_3'):
        n_out = 512
        x, params = fc(x, n_out, fc_init)
        tf.add_to_collection('classification_train', params)
        regularization(x, params, tf.get_variable_scope().name, FC_WEIGHT_DECAY)
        # x = bernouilly_activation(x, training_mode)
        x = activation(x, training_mode)
        x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.80), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('layer_2'):
        n_out = 512
        x, params = fc(x, n_out, fc_init)
        tf.add_to_collection('classification_train', params)
        regularization(x, params, tf.get_variable_scope().name, FC_WEIGHT_DECAY)
        # x = bernouilly_activation(x, training_mode)
        x = activation(x, training_mode)
        x = tf.cond(training_mode, lambda: tf.nn.dropout(x, 0.70), lambda: tf.nn.dropout(x, 1))

    with tf.variable_scope('layer_1'):
        outputs, params = fc(x, 10, fc_init)
        tf.add_to_collection('classification_train', params)
        # num_units_in = x.get_shape()[1]
        # num_units_out = 10
        # weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=fc_init, dtype='float')
        # biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer(), dtype='float')
        # params = [weights, biases]
        # S,U,V = tf.svd(weights, full_matrices=True)        
        # Uk = U[:,0:10]
        # yact = tf.matmul(tf.matmul(x,Uk),tf.transpose(Uk))
        # outputs = tf.nn.xw_plus_b(x, weights, biases) 

        reg = regularization(outputs, params, tf.get_variable_scope().name, FC_WEIGHT_DECAY)

    return outputs, [reg[0]]#, yact, x]
