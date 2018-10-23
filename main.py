import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm as tq
import time
import config
from worker_train import Worker_train as Worker

opts = config.init_config()

from CIFAR.dataset import Dataset
train_set = Dataset(opts, 'train')
val_set = Dataset(opts, 'val')


n_batch_per_eproch = 400
n_epoch = 10



worker = Worker(opts, train_set, val_set)

begin_time = time.time()
# for epoch in range(n_epoch):        
#         worker.run_epoch(n_batch_per_eproch)
#     # valid(worker)

worker.run_training()


end_time = time.time()
print(end_time - begin_time)
