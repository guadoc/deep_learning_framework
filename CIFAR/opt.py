import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='Framework for machine learning with Tensor Flow', formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('-en',       '--experience_name'  , default='TEST',                                    help='name of the experience')
    #model config    
    parser.add_argument('-mp',       '--model_path'       , default='CIFAR.models_cifar',                         help='model folder, must contain __init__ file')
    #data config
    parser.add_argument('-m',        '--model'            , default='resnet',                                     help='model file')
    parser.add_argument('-dp',       '--dataset'          , default='../datasets/cifar/dataset_cifar',                        help='folder with metadata configuration')
    parser.add_argument('-d',        '--data'             , default='../datasets/cifar/dataset_cifar/bin',                  help='folder with data')
    # parser.add_argument('-d',        '--data'             , default='/net/leodelibes/blot/cifar10/bin',                  help='folder with data')
    #saves and logs folders
    parser.add_argument('-le',       '--last_epoch'       , default=0,                                            help='last epoch, must load epoch model')
    parser.add_argument('-ind',      '--index'            , default=25,                                           help='folder with experiments log and variables, None for last')
    parser.add_argument('-ca',       '--cache'            , default='/net/leodelibes/blot/cifar/expe',               help='folder with experiments variables')
    parser.add_argument('-sm',       '--save_model'       , default=False,                                        help='Decides if model weights are saved after each epoch')
    parser.add_argument('-log',      '--log'              , default='./CIFAR/expe',                               help='folder with experiments logs')
    parser.add_argument('-sl',       '--save_logs'        , default=True,                                         help='Decides if training logs are saved after each batch/epoch')
    parser.add_argument('-il',       '--init_logs'        , default=False,                                         help='Decides if training logs are initialized')
    #training config
    parser.add_argument('-chkp',     '--checkpoint'       , default=1,                                            help='number of batch for each checkpoint')
    parser.add_argument('-ne',       '--n_epoch'          , default=550,                                          help='total number of epoch')
    parser.add_argument('-bs',       '--batch_size'       , default=128,                                          help='number of exemple per batch')
    parser.add_argument('-ntrain',   '--n_data_train'     , default=50000,                                        help='number of data in train set')
    parser.add_argument('-nval',     '--n_data_val'       , default=10000,                                        help='number of data in validation set')
    #not used yet
    parser.add_argument('-lo',       '--loss'             , default='logloss',                                    help='')
    parser.add_argument('-op',       '--optim'            , default='sgd',                                        help='')
    parser.add_argument('-reg',      '--regularizer'      , default='shade',                                      help='')
    parser.add_argument('-tl',       '--train_loaders'    , default=8,                                           help='')
    parser.add_argument('-vl',       '--val_loaders'      , default=8,                                           help='')
    args = parser.parse_args()    
    return args

