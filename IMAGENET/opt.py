import argparse
import os

def parse_config():
    parser = argparse.ArgumentParser(description='Framework for machine learning with Tensor Flow', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-en',       '--experience_name'  , default='testdistribute',                                        help='name of the experience')
    parser.add_argument('-mp',       '--model_path'       , default='IMAGENET.models_imagenet',                          help='model folder, must contain __init__ file')
    parser.add_argument('-m',        '--model'            , default='resnet101',                                         help='model file')

    parser.add_argument('-dp',       '--dataset'          , default='../datasets/imagenet/dataset_imagenet',                         help='folder with metadata configuration')
    parser.add_argument('-d',        '--data'             , default='/local/common-data/imagenet_2012/images',           help='folder with data')
    #parser.add_argument('-d',        '--data'             , default='/local/chenm/data/imagenet',                        help='folder with data')
    parser.add_argument('-le',       '--last_epoch'       , default=0,                                                   help='last epoch, must load epoch model')
    parser.add_argument('-ind',      '--index'            , default=None,                                                  help='folder with experiments log and variables, None for last')
    parser.add_argument('-ca',       '--cache'            , default='/net/leodelibes/blot/imagenet/expe',                   help='folder with experiments variables')
    parser.add_argument('-sm',       '--save_model'       , default=False,                                               help='Decide if model weights are saved after each epoch')
    parser.add_argument('-log',      '--log'              , default='/net/leodelibes/blot/imagenet/expe',                   help='folder with experiments logs')
    parser.add_argument('-sl',       '--save_logs'        , default=False,                                               help='Decide if training logs are saved after each batch/epoch')
    parser.add_argument('-il',       '--init_logs'        , default=False,                                               help='Decide if training logs are saved after each batch/epoch')
    #training config
    parser.add_argument('-ne',       '--n_epoch'          , default=100,                                                 help='total number of epoch')
    parser.add_argument('-bs',       '--batch_size'       , default=16,                                                  help='number of exemple per batch')
    parser.add_argument('-chkp',     '--checkpoint'       , default=10,                                                  help='number of batch for each checkpoint')
    parser.add_argument('-ntrain',   '--n_data_train'     , default=1200000,                                                   help='number of data in train set')
    parser.add_argument('-nval',     '--n_data_val'       , default=50000,                                                   help='number of data in validation set')
    # not in use yet
    parser.add_argument('-op',       '--optim'            , default='sgd',                                               help='number of thread feeding queue during training')
    parser.add_argument('-tl',       '--train_loaders'    , default=16,                                                  help='number of thread feeding queue during training')
    parser.add_argument('-vl',       '--val_loaders'      , default=16,                                                  help='number of thread feeding queue during validation')
    args = parser.parse_args()    
    return args


