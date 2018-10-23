import os
import shutil
import sys



def init_config(args):            
    args.index = args.index or index_max(args.log)    
    print("### Experience indexed: %d"%(args.index))
    folder_log = os.path.join(args.log, str(args.index) + "_" + args.model+ '_'+args.experience_name)
    folder_cache = os.path.join(args.cache, str(args.index) + "_" + args.model+ '_'+args.experience_name)
    if args.save_logs:
        try:
            os.stat(folder_log)
        except:
            print(folder_log)
            os.mkdir(folder_log)        
    if args.save_model:
        try:
            os.stat(folder_cache)
        except:
            os.mkdir(folder_cache)    
    args.cache = os.path.join(folder_cache, "models")
    args.log = os.path.join(folder_log, "logs")
    if args.save_logs:
        try:
            os.stat(args.log)
        except:
            os.mkdir(args.log)
        if args.init_logs:
            shutil.rmtree(args.log)
            os.mkdir(args.log)
    if args.save_model:
        try:
            os.stat(args.cache)
        except:
            os.mkdir(args.cache)
    print('### Logs  path is ' + args.log)
    print('### Cache path is ' + args.cache)
    sys.path.append(args.dataset)
    return args


def index_max(path):
    index = 0
    for f in os.listdir(path):
        if not os.path.isfile(f):
            tab = f.split("_")
            ind = int(float(tab[0]))
            if ind > index:
                index=ind
    return index+1