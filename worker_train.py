
import numpy as np
import tensorflow as tf
import importlib

from monitor import Monitor
      
import math
import time
import tqdm as tq



class Worker_train():
    def __init__(self, opts, train_set, val_set, sess = None, init_W=None):
        self.graph = tf.Graph()
        self.sess = sess or tf.Session(graph=self.graph)          
        with self.graph.as_default():        
            self.init_model(opts, train_set, val_set)        
            self.launch_queus()
        

    def init_model(self, opts, train_set, val_set):
        architecture = importlib.import_module(opts.model_path+'.'+opts.model)     
        self.train_model = architecture.Model(opts, self.sess)        
        self.monitor = Monitor(opts, self.train_model, train_set, val_set)                                    
        self.train_model.initialize_parameters(opts)        
        

    def launch_queus(self):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)


    def get_W(self):
        with self.graph.as_default(): 
            return self.sess.run(tf.trainable_variables())


    def get_W_shapes():
        pass


    def run_batch_optim(self):              
        with self.graph.as_default():   
            optim_params = self.train_model.optim_param_schedule(self.monitor.board)
            optim, metrics = self.sess.run([ self.monitor.optim
                                            ,self.monitor.train_metrics 
                                            # ,self.monitor.board.batch_summary
                                            ],
                                        feed_dict={
                                            self.monitor.batch_size: self.monitor.train_batch_size,
                                            self.monitor.training_mode: True,
                                            self.monitor.lr: optim_params["lr"],
                                            self.monitor.momentum: optim_params["momentum"]
                                        })
            self.monitor.board.update_batch_data(metrics, self.monitor.train_batch_size)
        

    def run_epoch(self, n_batches):
        self.monitor.board.init_epoch()
        optim_params = self.train_model.optim_param_schedule(self.monitor.board)
        print("lr: "+str(optim_params['lr'])+ ", momentum: "+str(optim_params['momentum']) )
        for batch in tq.tqdm(range(n_batches)):
            self.run_batch_optim()
        self.monitor.board.end_epoch()
        if self.monitor.save_model:
            self.monitor.model.model_save(self.monitor.board.epoch)
        


    def run_training(self):
        n_batch = math.floor(self.monitor.train_set.size  / self.monitor.train_batch_size)
        while self.monitor.board.epoch < self.monitor.n_epoch:
            self.run_epoch(n_batch)
            self.run_validation()
            self.monitor.board.print_perfs()



    def run_validation(self):
        n_batch = math.floor(self.monitor.val_set.size  / self.monitor.val_batch_size)
        self.monitor.board.init_val_data()
        for batch in tq.tqdm(range(n_batch)):
            metrics = self.sess.run(self.monitor.val_metrics,
                                        feed_dict={
                                            self.monitor.batch_size: self.monitor.val_batch_size,
                                            self.monitor.training_mode: False}) 
            self.monitor.board.update_val_data(metrics, self.monitor.val_batch_size)



#############################################TO TEST#############################################


    def get_batches(self):
        with self.graph.as_default():                     
            for bat in range(5000):           
                time.sleep(0.01) 
                images = self.sess.run(self.monitor.inputs, feed_dict={self.monitor.batch_size: 10, self.monitor.training_mode: True})


    def run(self):
        # self.get_batch()        
        self.run_batch_optim()


        

