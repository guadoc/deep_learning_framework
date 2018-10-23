import numpy as np
import tensorflow as tf
import math
from scipy import stats as st
import os
import time
from tensorboardX import SummaryWriter


import matplotlib.pyplot as plt


class Board:
    def __init__(self, opts):
        self.epoch = opts.last_epoch or 0
        self.sample_epoch = 0
        self.batch_epoch = 0
        self.batch_total = 0

        self.save_logs = opts.save_logs
        if self.save_logs:
            self.log_writer = SummaryWriter(opts.log)   
            self.batch_summary_list = self.batch_summary()
            self.epoch_summary_list = self.epoch_summary()



    def batch_summary(self):
        summary_list = {'classification_loss'}        
        return summary_list


    def epoch_summary(self):
        summary_list =  {'train_epoch_classification_loss'
                        ,'val_epoch_classification_loss'
                        ,'train_epoch_accuracy_top1'
                        ,'val_epoch_accuracy_top1'
                        ,'train_epoch_accuracy_top5'
                        ,'val_epoch_accuracy_top5'
                        }
        return summary_list

    
    def update_val_data(self, metrics, batch_size):
        self.batch_val +=1
        self.sample_val += batch_size        
        for metric_name, metric_value in self.val_cumul_metrics.items():
            self.val_cumul_metrics[metric_name] += metrics[metric_name]        

    def init_val_data(self):
        self.batch_val = 0
        self.sample_val = 0
        self.val_cumul_metrics = {
            'classification_loss':0,
            'accuracy_top1':0, 
            'accuracy_top5':0
        }

    def update_batch_data(self, metrics, batch_size):
        self.batch_epoch += 1
        self.batch_total += 1        
        self.sample_epoch += batch_size
        for metric_name, metric_value in self.train_epoch_cumul_metrics.items():
            self.train_epoch_cumul_metrics[metric_name] += metrics[metric_name]
        if self.save_logs:
            for metric_name in self.batch_summary_list:
                self.log_writer.add_scalar(metric_name, metrics[metric_name], self.batch_total)

    def init_epoch(self):
        self.train_epoch_cumul_metrics = {
            'classification_loss':0,
            'accuracy_top1':0, 
            'accuracy_top5':0
        }
        self.begin_epoch_time = time.time()
        self.batch_epoch = 0
        self.sample_epoch = 0

    def end_epoch(self):
        self.epoch+=1


    def print_perfs(self):               
        m, s = divmod(time.time() - self.begin_epoch_time, 60)
        h, m = divmod(m, 60)
        epoch_metrics = {}
        epoch_metrics['train_epoch_accuracy_top1'] = 100*self.train_epoch_cumul_metrics['accuracy_top1']/self.sample_epoch
        epoch_metrics['train_epoch_accuracy_top5'] = 100*self.train_epoch_cumul_metrics['accuracy_top5']/self.sample_epoch        
        epoch_metrics['train_epoch_classification_loss'] = self.train_epoch_cumul_metrics['classification_loss']/self.batch_epoch
        epoch_metrics['val_epoch_accuracy_top1'] = 100*self.val_cumul_metrics['accuracy_top1']/self.sample_val
        epoch_metrics['val_epoch_accuracy_top5'] = 100*self.val_cumul_metrics['accuracy_top5']/self.sample_val
        epoch_metrics['val_epoch_classification_loss'] = self.val_cumul_metrics['classification_loss']/self.batch_val        


        if self.save_logs:
            for metric_name in self.epoch_summary_list:
                self.log_writer.add_scalar(metric_name, epoch_metrics[metric_name], self.epoch)

        print('Epoch %d[%d:%d:%d], Test[loss: %.3f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%], Train[loss: %.4f, accuracy(top1): %.2f%%, accuracy(top5): %.2f%%]'%\
                (self.epoch,h, m, s
                ,epoch_metrics['val_epoch_classification_loss']
                ,epoch_metrics['val_epoch_accuracy_top1']
                ,epoch_metrics['val_epoch_accuracy_top5']
                ,epoch_metrics['train_epoch_classification_loss']
                ,epoch_metrics['train_epoch_accuracy_top1']
                ,epoch_metrics['train_epoch_accuracy_top5']))

            


