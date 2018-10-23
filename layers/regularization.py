
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops


def weight_decay(outputs, params):
    reg = 0.    
    for w in params:
        reg += tf.nn.l2_loss(w)        
    return reg



def noize(outputs, params):    
    noize = tf.random_normal(params[0].get_shape(), mean=0, stddev=1)
    reg = tf.reduce_sum(tf.multiply(noize, params[0])) 
    return reg



def shade_old(outputs, params, collection, beta):
    REG_COEF = 0.9
    EXPOSANT = 2#0.2
    n_units = outputs.get_shape()[-1]
    # moving_mean_ = tf.get_variable('moving_mean_', [n_units], initializer=tf.constant_initializer(1), trainable=False)
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.constant_initializer(1), trainable=False)    
    moving_mean_0 = tf.get_variable('moving_mean_0', [n_units], initializer=tf.constant_initializer(-1), trainable=False)
    # sum_ = tf.reduce_sum(tf.nn.relu(outputs), axis=list(range(len(outputs.shape)-1)))
    # number_ = tf.count_nonzero(tf.nn.relu(outputs), axis=list(range(len(outputs.shape)-1)), dtype=tf.float32) + 0.0005
    # moving_mean_ = moving_averages.assign_moving_average(moving_mean_, sum_/number_, REG_COEF)
    # moving_mean_ = control_flow_ops.cond(number_ > 0, lambda: moving_averages.assign_moving_average(moving_mean_, sum_/number_, REG_COEF), lambda: moving_mean_)
    p_z1 = tf.get_variable('p_z1', [n_units], initializer=tf.constant_initializer(0.5), trainable=False)
    p_mode_1 = 1 - tf.exp(-tf.nn.relu(outputs))
    # p_mode_1 = tf.sigmoid(outputs - moving_mean_)
    p_z1 = moving_averages.assign_moving_average(p_z1, tf.reduce_mean(p_mode_1, [0]), REG_COEF, zero_debias=False)
    moving_mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), p_z1), REG_COEF, zero_debias=False)
    moving_mean_0 = moving_averages.assign_moving_average(moving_mean_0, tf.divide(tf.reduce_mean(tf.multiply(1-p_mode_1, outputs), [0]), 1-p_z1), REG_COEF, zero_debias=False)    
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.pow(tf.abs(tf.subtract(outputs, moving_mean_1) ), EXPOSANT), p_mode_1) ,[0])  )
    var_0 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.pow(tf.abs(tf.subtract(outputs, moving_mean_0) ), EXPOSANT), 1-p_mode_1) ,[0])  )
    reg = (var_1 + var_0)
    for w in params:
        # reg += tf.nn.l2_loss(w)
        tf.add_to_collection(collection+"_variables", w)    
    tf.add_to_collection(collection+'_reg', tf.multiply(reg, beta, name='reg'))        
    return reg

def shade(outputs, params):
    REG_COEF = 0.9
    EXPOSANT = 2
    n_units = outputs.get_shape()[1]
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.constant_initializer(1), trainable=False)    
    moving_mean_0 = tf.get_variable('moving_mean_0', [n_units], initializer=tf.constant_initializer(-1), trainable=False)
    p_z1 = tf.get_variable('p_z1', [n_units], initializer=tf.constant_initializer(0.5), trainable=False)
    p_mode_1 = 1 - tf.exp(-tf.nn.relu(outputs))
    p_z1 = moving_averages.assign_moving_average(p_z1, tf.reduce_mean(p_mode_1, [0]), REG_COEF, zero_debias=False)
    moving_mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), p_z1), REG_COEF, zero_debias=False)
    moving_mean_0 = moving_averages.assign_moving_average(moving_mean_0, tf.divide(tf.reduce_mean(tf.multiply(1-p_mode_1, outputs), [0]), 1-p_z1), REG_COEF, zero_debias=False)    
    var_1 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.pow(tf.abs(tf.subtract(outputs, moving_mean_1) ), EXPOSANT), p_mode_1) ,[0])  )
    var_0 = tf.reduce_sum(  tf.reduce_mean(tf.multiply( tf.pow(tf.abs(tf.subtract(outputs, moving_mean_0) ), EXPOSANT), 1-p_mode_1) ,[0])  )
    reg = (var_1 + var_0)     
    return reg


def shade_conv(outputs, params):
    shape_output = outputs.get_shape()
    sizeX = shape_output[-1]
    sizeY = shape_output[-2]
    indexX = tf.random_uniform([], minval=0, maxval=sizeX, dtype=tf.int32 )
    indexY = tf.random_uniform([], minval=0, maxval=sizeY, dtype=tf.int32 )
    # outs = outputs[:, :, indexX, indexY]
    outs = outputs[:, indexX, indexY, :]
    return shade(outs, params)


def shade_kernel(outputs, params):
    REG_COEF = 0.9
    EXPOSANT = 2
    x = outputs
    n_units = outputs.get_shape()[-1]    
    moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.constant_initializer(1), trainable=False)    
    moving_mean_0 = tf.get_variable('moving_mean_0', [n_units], initializer=tf.constant_initializer(-1), trainable=False)
    p_z1 = tf.get_variable('p_z1', [n_units], initializer=tf.constant_initializer(0.5), trainable=False)
    p_mode_1 = tf.get_variable('p_mode1', [n_units], initializer=tf.constant_initializer(0.5), trainable=False)
    S,U,V = tf.svd(params[0], full_matrices=True)            
    Uk = U[:,0:10]
    yact = tf.matmul(tf.matmul(x,Uk),tf.transpose(Uk))
    p_mode_1 = tf.sigmoid(yact)
    p_mode_1 = tf.stop_gradient(p_mode_1)
    # p_mode_1 = tf.Print(p_mode_1, [tf.reduce_mean(p_mode_1)])
    p_z1 = moving_averages.assign_moving_average(p_z1, tf.reduce_mean(p_mode_1, [0]), REG_COEF, zero_debias=False)
    moving_mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, outputs), [0]), p_z1), REG_COEF, zero_debias=False)
    moving_mean_0 = moving_averages.assign_moving_average(moving_mean_0, tf.divide(tf.reduce_mean(tf.multiply(1-p_mode_1, outputs), [0]), 1-p_z1), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum( tf.reduce_mean(tf.multiply( tf.pow(tf.abs(tf.subtract(outputs, moving_mean_1) ), EXPOSANT), p_mode_1  ) ,[0]) )
    var_0 = tf.reduce_sum( tf.reduce_mean(tf.multiply( tf.pow(tf.abs(tf.subtract(outputs, moving_mean_0) ), EXPOSANT), 1-p_mode_1) ,[0]) )
    reg = (var_1 + var_0)     
    return [reg]


def log(x):    
    return tf.log(tf.clip_by_value(x,1e-7,1e7))

def normal_density(x, mean, var):
    return tf.divide( tf.exp(-tf.divide(tf.square(x-mean),2*var)) , tf.sqrt(2*np.pi*var) )

def reve(outputs, params, beta, labels): ################################## \Ker vshade
    N_SAMPLE = 12        
    W = params[0]
    S,U,V = tf.svd(W)
    Uk = tf.stop_gradient(U)
    yproj = tf.matmul(tf.matmul(outputs,Uk),tf.transpose(Uk))
    p_mode_1 = tf.sigmoid(yproj)
    alpha = tf.reduce_mean(p_mode_1, [0])
    mean_1 = tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, yproj), [0]), alpha)
    mean_0 = tf.divide(tf.reduce_mean(tf.multiply(1-p_mode_1, yproj), [0]), 1-alpha)
    var_1 = tf.divide( tf.reduce_mean(tf.multiply(p_mode_1,tf.square(tf.subtract(yproj,mean_1))), [0]), alpha )
    var_0 = tf.divide( tf.reduce_mean(tf.multiply(1-p_mode_1,tf.square(tf.subtract(yproj,mean_0))), [0]), 1-alpha )
    
    shape = outputs.get_shape().as_list()
    n_units = shape[1:]
    STDDEV = 0.01
    sum_reg1 = tf.constant(0.0)
    sum_reg2 = tf.constant(0.0)
    sum_reg3 = tf.constant(0.0)
    for i in range(N_SAMPLE):
        eps = tf.random_normal(n_units,stddev=STDDEV)        
        z = tf.add(outputs,eps)
        z = tf.matmul(tf.matmul(z,Uk),tf.transpose(Uk))        
        with tf.variable_scope('classifier', reuse=True):
            classif = tf.nn.xw_plus_b(z, params[0], params[1]) 
        sum_reg1 -= tf.reduce_sum(tf.reduce_mean(tf.multiply(log(tf.nn.softmax(classif)), tf.one_hot(labels,classif.get_shape()[-1])),[0] )) / (N_SAMPLE)#*336)
        sum_reg2 -= tf.reduce_mean( log( alpha*normal_density(z, mean_1, var_1)+(1-alpha)*normal_density(z,mean_0,var_0)) ) / N_SAMPLE
        # sum_reg2 -= tf.reduce_sum( log( alpha*normal_density(z, mean_1, var_1)+(1-alpha)*normal_density(z,mean_0,var_0)) ) / N_SAMPLE    
    return sum_reg1 + beta*sum_reg2#, sum_reg1, sum_reg2, sum_reg3, mean_1[0], mean_0[0], var_1[0], var_0[0], alpha[0]





def variational_shade_weldone(outputs, params, beta, labels): ################################## \Ker vshade
    N_SAMPLE = 12        
    W = params[0][0,0,:,:]
    S,U,V = tf.svd(W)
    Uk = tf.stop_gradient(U)

    # yproj = tf.matmul(tf.matmul(outputs,Uk),tf.transpose(Uk))
    Uconv = tf.expand_dims(tf.expand_dims(Uk, 0), 0)
    Uconvt = tf.expand_dims(tf.expand_dims(tf.transpose(Uk), 0), 0)


    Y1 = tf.nn.conv2d(outputs, Uconv, [1,1,1,1], padding = 'VALID', data_format='NCHW')
    Y2 = tf.nn.conv2d(Y1, Uconvt, [1,1,1,1], padding = 'VALID', data_format='NCHW')    
    
    p_mode_1 = tf.sigmoid(Y2)
    alpha = tf.reduce_mean(p_mode_1, [0])
    mean_1 = tf.divide(tf.reduce_mean(tf.multiply(p_mode_1, Y2), [0]), alpha)
    mean_0 = tf.divide(tf.reduce_mean(tf.multiply(1-p_mode_1, Y2), [0]), 1-alpha)
    var_1 = tf.divide( tf.reduce_mean(tf.multiply(p_mode_1,tf.square(tf.subtract(Y2,mean_1))), [0]), alpha )
    var_0 = tf.divide( tf.reduce_mean(tf.multiply(1-p_mode_1,tf.square(tf.subtract(Y2,mean_0))), [0]), 1-alpha )
    
    shape = outputs.get_shape().as_list()
    n_units = shape[1:]
    STDDEV = 0.1
    sum_reg1 = tf.constant(0.0)
    sum_reg2 = tf.constant(0.0)
    sum_reg3 = tf.constant(0.0)
    for i in range(N_SAMPLE):
        eps = tf.random_normal(n_units,stddev=STDDEV)        
        Z = tf.add(outputs,eps)
        Z1 = tf.nn.conv2d(Z, Uconv, [1,1,1,1], padding = 'VALID', data_format='NCHW')
        Z2 = tf.nn.conv2d(Z1, Uconvt, [1,1,1,1], padding = 'VALID', data_format='NCHW')          
        OUT = tf.nn.conv2d(Z2, params[0], [1,1,1,1], padding = 'VALID', data_format='NCHW')        
        # OUT = tf.reshape(OUT, [- 1, 1000, 14*14])            
        # sorted_val, ind = tf.nn.top_k(  OUT,    k=196,    sorted=True)    
        # OUTPUTS = (tf.reduce_sum(sorted_val[:, :, 0:50], axis=2) + tf.reduce_sum(sorted_val[:, :, 146:], axis=2))/50
        OUT = tf.reshape(OUT, [- 1, 10, 7*7])            
        sorted_val, ind = tf.nn.top_k(  OUT,    k=49,    sorted=True)    
        OUTPUTS = (tf.reduce_sum(sorted_val[:, :, 0:22], axis=2) + 0.6*tf.reduce_sum(sorted_val[:, :, 38:], axis=2))/33

        sum_reg1 -= tf.reduce_sum(tf.reduce_mean(tf.multiply(log(tf.nn.softmax(OUTPUTS)), tf.one_hot(labels,OUTPUTS.get_shape()[-1])),[0] )) / (N_SAMPLE)#*336)
        sum_reg2 -= tf.reduce_mean( log( alpha*normal_density(Z2, mean_1, var_1)+(1-alpha)*normal_density(Z2,mean_0,var_0)) ) / N_SAMPLE
        # sum_reg2 -= tf.reduce_sum( log( alpha*normal_density(z, mean_1, var_1)+(1-alpha)*normal_density(z,mean_0,var_0)) ) / N_SAMPLE    
    return sum_reg1 + beta*sum_reg2#, sum_reg1, sum_reg2, sum_reg3, mean_1[0], mean_0[0], var_1[0], var_0[0], alpha[0]





def var_entropy(outputs, params, collection, beta):
    REG_COEF = 0.8
    n_units = outputs.get_shape()[-1]
    moving_mean_1 = tf.get_variable('moving_mean_reg', [n_units], initializer=tf.zeros_initializer(), trainable=False)
    mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.reduce_mean(outputs, [0]), REG_COEF, zero_debias=False)
    var_1 = tf.reduce_sum(tf.reduce_mean( tf.square( tf.subtract(outputs, mean_1) ) ,[0])  )
    for weights in params:
        tf.add_to_collection(collection+"_variables", weights)
    tf.add_to_collection(collection+'_reg', tf.multiply(var_1, beta, name='reg'))
    return var_1



def var_entropy_conv(outputs, params, collection, beta):
    shape_output = outputs.get_shape()
    sizeX = shape_output[1]
    sizeY = shape_output[2]
    indexX = tf.random_uniform([], minval=0, maxval=sizeX, dtype=tf.int32 )
    indexY = tf.random_uniform([], minval=0, maxval=sizeY, dtype=tf.int32 )
    outs = outputs[:, indexX, indexY, :]
    return var_entropy(outs, params, collection, beta)



