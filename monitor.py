import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.python.ops import control_flow_ops

from board import Board


class Monitor:
    def __init__(self, opts, model, train_set, val_set):            
        ###### model to train
        self.model = model
        self.save_model = opts.save_model

        ###### datasets
        self.train_set = train_set
        self.val_set = val_set

        ###### monitoring params    
        self.n_epoch = opts.n_epoch    
        self.train_batch_size = opts.batch_size or 128
        self.val_batch_size = opts.batch_size or 128
        
        ###### training params
        self.training_mode = tf.placeholder(tf.bool, shape=[])        
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        ###### sampling
        val_batch_size, train_batch_size = control_flow_ops.cond(self.training_mode, 
            lambda: (tf.constant(0), self.batch_size),
            lambda: (self.batch_size, tf.constant(0))
            )
        train_sample = self.train_set.sample(train_batch_size)
        val_sample = self.val_set.sample(val_batch_size)
        sample = control_flow_ops.cond(self.training_mode, lambda: train_sample, lambda: val_sample)
        self.inputs = sample["inputs"] #tf.constant(10.0, shape=[10, 3, 28, 28], dtype="float32")
        self.labels = sample["labels"] #tf.constant(0, shape=[10], dtype="int32")

        ####### inference
        self.outputs, self.losses, self.info = self.model.inference(self.inputs, self.labels, self.training_mode)
        self.train_metrics = self.train_metrics()
        self.val_metrics = self.val_metrics()        

        ####### optim           
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self.set_optimizer(opts)
            self.optim = self.loss_list_minimization()
                    

        ####### board 
        self.board = Board(opts)        
        



    def train_metrics(self):
        train_metrics =  {                        
                        'classification_loss':self.losses['classification_loss'],
                        'accuracy_top1':tf.reduce_sum(tf.cast(tf.equal(
                                tf.cast(tf.argmax(self.outputs,1), tf.int32),
                                tf.cast(self.labels, tf.int32))
                            , tf.float32)),   
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   self.outputs, self.labels, k=5 )  ))
                        }
        return train_metrics



    def val_metrics(self):
        val_metrics =  {                        
                        'classification_loss':self.losses['classification_loss'],                        
                        'accuracy_top1':tf.reduce_sum(tf.cast(
                            tf.equal(
                                tf.cast(tf.argmax(self.outputs,1), tf.int32),
                                tf.cast(self.labels, tf.int32))
                        , tf.float32)),                 
                        'accuracy_top5':tf.reduce_sum(tf.to_float(  tf.nn.in_top_k(   self.outputs, self.labels, k=5 )  ))   
                        }
        return val_metrics
    


    def loss_list_minimization(self):
        losses_optim_ = []
        for name, loss in self.losses.items():
            losses_optim_.append(self.optimizer.minimize(loss , var_list=tf.get_collection(name)))
        return losses_optim_



    def set_optimizer(self, opts):
        if opts.optim =='adam':
            print("## adam optim")            
            param = {"learning_rate":self.lr}
            return tf.train.AdamOptimizer(**param)
        elif opts.optim =='sgd':
            print("## sgd optim")
            self.lr       = tf.placeholder(tf.float32, shape=[])
            self.momentum = tf.placeholder(tf.float32, shape=[])
            param = {"learning_rate":self.lr, "momentum":self.momentum}
            return tf.train.MomentumOptimizer(**param)


