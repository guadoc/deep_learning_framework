import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

class Abstract_Dataset(ABC):
    def __init__(self, opts, data_type):    
        self.data_type = data_type                
        if self.data_type == 'train':
            self.nloader =  opts.train_loaders  
        else:
            self.nloader = opts.val_loaders
        self.init_metadata(data_type)
        self.input_shape = [self.input_depth, self.input_height, self.input_width]



    @abstractmethod
    def init_metadata(self, data_type):
        self.size = 10
        self.input_depth = 3
        self.input_height = 32
        self.input_width = 32


    @abstractmethod
    def build_input(self):
        return {'inputs':tf.constant(1, shape=self.input_shape, dtype="float32"), 'labels':tf.constant(0, shape=[], dtype="int32")}


    def sample(self, batch_size):        
        if self.data_type == 'train':
            sample_queue = tf.RandomShuffleQueue(
                capacity= 10000,
                min_after_dequeue= 128,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.input_shape, [1]])

        elif self.data_type == 'val':         
            sample_queue = tf.FIFOQueue(
                capacity=1000,
                dtypes=[tf.float32, tf.int32],
                shapes=[self.input_shape, [1]])
        else:
            pass
        image, label = self.build_input()
        sample_enqueue_op = sample_queue.enqueue([image, label])
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(sample_queue, [sample_enqueue_op] * self.nloader))
        images, labels = sample_queue.dequeue_many(batch_size)            
        labels = tf.reshape(labels, [batch_size])    
        return {"inputs":images, "labels":labels}


