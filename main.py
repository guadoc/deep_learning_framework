import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm as tq
import time
import config
from worker_train import Worker_train as Worker


# sys.path.append('CIFAR')
sys.path.append( 'IMAGENET')

from opt import parse_config
args = parse_config()    
opts = config.init_config(args)


from dataset import Dataset
train_set = Dataset(opts, 'train')
val_set = Dataset(opts, 'val')

worker = Worker(opts, train_set, val_set)

begin_time = time.time()
worker.run_training()
end_time = time.time()
print(end_time - begin_time)

