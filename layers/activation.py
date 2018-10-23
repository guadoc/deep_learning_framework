import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops

def relu(x, is_training):
    return tf.nn.relu(x)

def lrelu(x, alpha, is_training):
    return tf.maximum(alpha*x, x)


def bernouilly_activation(x, is_training):
    decay = 0.99
    x_shape = x.get_shape().as_list()
    moving_mean = tf.get_variable('moving_activation_mean', [x_shape[-1]], initializer=tf.ones_initializer(), trainable=False)        
    sum_ = tf.reduce_sum(tf.nn.relu(x), axis=list(range(len(x.shape)-1)))
    number_ = tf.count_nonzero(tf.nn.relu(x), axis=list(range(len(x.shape)-1)), dtype=tf.float32) +0.0005        
    moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_mean, sum_/number_, decay), lambda: moving_mean)
    return (tf.divide(tf.sign(x-moving_mean)+1, 2))*moving_mean  
    #return (tf.divide(tf.sign(x)+1, 2))



def input_binary_activation(x, is_training):
    decay = 0.99
    x_shape = x.get_shape().as_list()
    pos_moving_mean = tf.get_variable('pos_moving_activation_mean', [x_shape[-1]], initializer=tf.constant_initializer(1), trainable=False)        
    neg_moving_mean = tf.get_variable('neg_moving_activation_mean', [x_shape[-1]], initializer=tf.constant_initializer(-1), trainable=False)        
    moving_mean = tf.get_variable('moving_activation_mean', [x_shape[-1]], initializer=tf.constant_initializer(0), trainable=False)        
    
    pos_sum_ = tf.reduce_sum(tf.nn.relu(x- moving_mean), axis=list(range(len(x.shape)-1)))    
    neg_sum_ = -tf.reduce_sum(tf.nn.relu(-x+ moving_mean), axis=list(range(len(x.shape)-1)))        

    pos_number_ = tf.count_nonzero(tf.nn.relu(x- moving_mean), axis=list(range(len(x.shape)-1)), dtype=tf.float32) +0.0005        
    neg_number_ = tf.count_nonzero(tf.nn.relu(-x+ moving_mean), axis=list(range(len(x.shape)-1)), dtype=tf.float32) +0.0005        
    # neg_number_ = x_shape[0] - pos_number_
    # neg_number_ = tf.Print(neg_number_, [neg_number_ + pos_number_])    

    pos_moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(pos_moving_mean, pos_sum_/pos_number_, decay), lambda: pos_moving_mean)        
    neg_moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(neg_moving_mean, neg_sum_/neg_number_, decay), lambda: neg_moving_mean)        
    moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_mean, tf.reduce_mean(x, axis=list(range(len(x.shape)-1))), decay), lambda: moving_mean)        
    # pos_moving_mean = tf.Print(pos_moving_mean, [tf.reduce_mean((tf.divide(tf.sign(x)+1, 2))*pos_moving_mean), tf.reduce_mean((tf.divide(tf.sign(-x)+1, 2))*neg_moving_mean)])
    return (tf.divide(tf.sign(x)+1, 2))*pos_moving_mean + (tf.divide(tf.sign(-x)+1, 2))*neg_moving_mean



def stochastic_bernouilli_activation(outputs, is_training):
    # REG_COEF = 0.9
    # EXPOSANT = 2#0.2
    # n_units = outputs.get_shape()[-1]
    # biases = tf.get_variable('activ_biases', [n_units], initializer=tf.constant_initializer(0), trainable=True)
    x = outputs #- biases
    # moving_mean_1 = tf.get_variable('moving_mean_1', [n_units], initializer=tf.constant_initializer(1), trainable=False)
    # moving_mean_0 = tf.get_variable('moving_mean_0', [n_units], initializer=tf.constant_initializer(-1), trainable=False)
    # p_z1 = tf.get_variable('p_z1', [n_units], initializer=tf.constant_initializer(0.5), trainable=False)    
    p_mode_1 = 1 - tf.exp(-tf.nn.relu(x))
    #p_z1 = moving_averages.assign_moving_average(p_z1, tf.reduce_mean(p_mode_1, axis=list(range(len(x.shape)-1))), REG_COEF, zero_debias=False)
    z = tf.distributions.Bernoulli(probs=p_mode_1, dtype=tf.float32)
    Z = z.sample()
    
    # m_Z1 = tf.reduce_sum(tf.multiply(p_mode_1, x), axis=list(range(len(x.shape)-1))) / (tf.reduce_sum(p_mode_1, axis=list(range(len(x.shape)-1))) +0.00005)
    # m_Z0 = tf.reduce_sum(tf.multiply(1-p_mode_1, x), axis=list(range(len(x.shape)-1))) / (tf.reduce_sum(1 - p_mode_1, axis=list(range(len(x.shape)-1))) +0.00005)
    # m_Z1 = tf.reduce_sum(x*Z, axis=list(range(len(x.shape)-1)))  / (tf.count_nonzero(x*Z, axis=list(range(len(x.shape)-1)), dtype=tf.float32)  +0.0001)
    # m_Z0 = tf.reduce_sum(x*(1-Z), axis=list(range(len(x.shape)-1)))  /  (tf.count_nonzero(x*(1-Z), axis=list(range(len(x.shape)-1)), dtype=tf.float32)   +0.0001)

    # m_0 = tf.Print(m_0, ["m_0", tf.shape(m_0)])
    # mean_1 = moving_averages.assign_moving_average(moving_mean_1, tf.divide(m_Z1, p_z1), REG_COEF, zero_debias=False)    
    # mean_0 = moving_averages.assign_moving_average(moving_mean_0, tf.divide(m_Z0, 1-p_z1), REG_COEF, zero_debias=False)     
    # mean_1 = moving_averages.assign_moving_average(moving_mean_1, m_Z1, REG_COEF, zero_debias=False)    
    # mean_0 = moving_averages.assign_moving_average(moving_mean_0, m_Z0, REG_COEF, zero_debias=False)     
    # mean_0 = tf.Print(mean_0, ["mean_0", tf.shape(mean_0)])
    # update_moving_mean = control_flow_ops.cond(is_training, lambda: moving_averages.assign_moving_average(moving_mean, mean, momentum), lambda: moving_mean)
    # m_0 = control_flow_ops.cond(is_training, lambda: m_Z0, lambda: moving_mean_0)
    # m_1 = control_flow_ops.cond(is_training, lambda: m_Z1, lambda: moving_mean_1)
    # with tf.control_dependencies([update_moving_mean, update_moving_variance]):
    #     return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon), [beta, gamma]
    x = control_flow_ops.cond(is_training, lambda: (1-Z)*x, lambda: x)
    # x = (1-Z)*x
    # x = tf.Print(x, ["x", tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])
    return tf.nn.relu(x)

