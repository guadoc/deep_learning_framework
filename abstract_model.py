import tensorflow as tf
import os

from abc import ABC, abstractmethod

class Abstract_Model(ABC):
    def __init__(self, opts, session=None):
        print('-- Loading model %s'%(opts.model))        
        # creation of a session with memory properties
        # config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        # config.gpu_options.allow_growth=True
        self.sess = session or tf.Session(config=config)
        self.saving_file = os.path.join(opts.cache, opts.model)


    def initialize_parameters(self, opts):        
        self.sess.run(tf.variables_initializer(self.get_parameters()))
        self.saver = tf.train.Saver(tf.trainable_variables() + tf.get_collection('variable_to_save'), max_to_keep=100)                
        if opts.last_epoch > 0:
            self.model_load(self.saving_file + "_" + str(opts.last_epoch) + ".ckpt")
        print('### Model %s initialized with %d parameters'%(opts.model, self.count_parameters()))

    
    def get_parameters(self):
        return tf.global_variables()


    def count_parameters(self):
        tot_nb_params = 0
        def get_nb_params_shape(shape):
            nb_params = 1
            for dim in shape:
                nb_params = nb_params*int(dim)
            return nb_params
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape()
            current_nb_params = get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params


    def model_save(self, epoch):
        path = self.saving_file + "_" + str(epoch) + ".ckpt"
        save_path = self.saver.save(self.sess, path)
        print("### Model saved in file: %s" % save_path)


    def model_load(self, path):
        print("-- Loading model from file: %s" % path)
        self.saver.restore(self.sess, path)
        print("### Model loaded from file: %s" % path)



    @abstractmethod
    def inference(self):
        return 0, 0, 0



    @abstractmethod
    def optim_param_schedule(self, board):
        return {"lr": 0.01, 'momentum': 0.9}



    