
import tensorflow as tf
import os
import math
from abstract_dataset import Abstract_Dataset


class Dataset(Abstract_Dataset):
    def __init__(self, opts, data_type):
        Abstract_Dataset.__init__(self, opts, data_type)
        self.datapath = opts.data



    def init_metadata(self, data_type):
        self.input_depth = 3
        self.input_height = 28
        self.input_width = 28
        self.image_size = 32
        if data_type == 'train':
            self.size =  50000 
        else:
            self.size = 10000



    def build_input(self):
        self.loading_image_size = 32
        dataset = 'cifar10'
        if dataset == 'cifar10':
            label_bytes = 1
            label_offset = 0
            num_classes = 10
        elif dataset == 'cifar100':
            label_bytes = 1
            label_offset = 1
            num_classes = 100
        else:
            raise ValueError('Not supported dataset %s', dataset)
        image_bytes = self.loading_image_size * self.loading_image_size * self.input_depth
        record_bytes = label_bytes + label_offset + image_bytes                  
        if self.data_type == 'train':
            # self.datapath = self.datapath+str(self.thread_id)
            print('Training data path is ' + self.datapath)            
            if dataset == 'cifar10':
                data_files = [
                #os.path.join(self.datapath, 'data_4000.bin'),
                os.path.join(self.datapath, 'data_batch_1.bin'),
                os.path.join(self.datapath, 'data_batch_2.bin'),
                os.path.join(self.datapath, 'data_batch_3.bin'),
                os.path.join(self.datapath, 'data_batch_4.bin'),
                os.path.join(self.datapath, 'data_batch_5.bin')
                ]
            elif dataset == 'cifar100':
                data_files = [os.path.join(self.datapath, 'cifar-100-binary/train.bin')]
        else:
            print('Validation data path is ' + self.datapath)
            if dataset == 'cifar10':
                data_files = [os.path.join(self.datapath, 'test_batch.bin')]
            elif dataset == 'cifar100':
                data_files = [os.path.join(self.datapath, 'cifar-100-binary/test.bin')]
        
        file_queue = tf.train.string_input_producer(data_files, shuffle=(self.data_type == 'train'))
        # Read examples from files in the filename queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(file_queue)
        # Convert these examples to dense labels and processed images.
        record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
        # Convert from string to [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
                           [self.input_depth, self.loading_image_size, self.loading_image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)        

        if self.data_type == 'train':
            image = tf.image.resize_image_with_crop_or_pad(image, self.image_size, self.image_size)
            image = tf.random_crop(image, [self.input_height, self.input_width, self.input_depth])
            image = tf.image.random_flip_left_right(image)        
        elif self.data_type == 'val':
            image = tf.image.resize_image_with_crop_or_pad(image, self.input_height, self.input_width)

        image = tf.image.per_image_standardization(image)       
        image = tf.transpose(image, [2, 0, 1])
        return image, label


