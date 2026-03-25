import json
import time
import argparse
import shutil

import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data
from core.data import SEMISUP_DATASETS

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train, set_extra_arguments
from core.utils import set_config_file_precedence
from core.utils import update_parsed_values
from core.utils import validate_train_arguments
from core.utils import Trainer
from core.utils import seed

from gowal21uncovering.utils import WATrainer

# distributed imports:
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from core.utils.distributed import CustomGreweGPUEnv
from torch import distributed
import datetime

# weights and biases

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


def distributed_train(args):
    """
    Trains the model as defined by args, it expects args to be parsed and verified
    """

    #create myselfe

    
    env = CustomGreweGPUEnv()

    device = torch.device(f'cuda:{env.local_rank}')
    torch.cuda.set_device(env.local_rank)
    torch.distributed.init_process_group("nccl", init_method="env://",
                                            timeout=datetime.timedelta(seconds=3600),
                                            world_size=env.world_size,
                                            rank=env.rank)
    print("ENV:", env.__dict__)
    f = open(os.devnull, "w")
    if env.rank > 0:
        sys.stdout = f
        sys.stderr = f

    # To speed up training
    torch.backends.cudnn.benchmark = True

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
    sampler_path = os.path.join(LOG_DIR, 'sampler-state-last.pt')
    if env.rank==0:
        if os.path.exists(LOG_DIR) and not args.resume_path:
            print('No resume path given but logs exist, deleting prior run/log.')
            #assert False, 'No resume path given but logs exist, deleting prior run/log is possible if you remove this assertion'
            shutil.rmtree(LOG_DIR)
        os.makedirs(LOG_DIR, exist_ok=True)
        logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

        with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    info = get_data_info(DATA_DIR)

    # Adversarial Training
    seed(args.seed)
    if args.tau:
        if env.rank==0:
            print ('Using WA.')
        trainer = WATrainer(info, args, distributed=env)
    else:
        assert False, 'Distributed not yet implemented for wa=False'
        trainer = Trainer(info, args)
    last_lr = args.lr

   

    trainer.model = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model)
    trainer.init_optimizer(args.num_adv_epochs, args.pct_start)


    trainer.model = DistributedDataParallel(trainer.model, device_ids=[env.local_rank], gradient_as_bucket_view=True)
    trainer.wa_model = DistributedDataParallel(trainer.wa_model, device_ids=[env.local_rank], gradient_as_bucket_view=True)

    batchsize = 1
    print('Batchsize=',batchsize)
  
    with get_accelerator().device(env.local_rank):
        flops, macs, params = get_model_profile(model=trainer.model,
                                        input_shape=(batchsize, 3, 32, 32), 
                                args=None, # list of positional arguments to the model.
                                kwargs=None, # dictionary of keyword arguments to the model.
                                print_profile=True, # prints the model graph with the measured profile attached to each module
                                detailed=False, # print the detailed profile
                                module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                top_modules=1, # the number of top modules to print aggregated profile
                                warm_up=10, # the number of warm-ups before measuring the time of each module
                                as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                ignore_modules=None) # the list of modules to ignore in the profiling

        


def break_for_data_switch(args, epoch):
    if not args.one_epoch: return False
    if epoch == 7000: return True 
    if epoch%2000==0: return True
    return False

def update_result_row(csv_row, trainer, test_dataloader, train_dataloader, sampler_path, epoch = None):
    if sampler_path and distributed.get_rank()==0:
        trainer.save_sampler_state(train_dataloader, sampler_path) 
    print('\n**Getting metrics on test data**')
    csv_row['PGD40_test_acc'], csv_row['PGD40_test_loss'], csv_row['clean_test_acc'], csv_row['clean_test_loss'] = trainer.Linf_PGD_40(test_dataloader, 'CE')
    csv_row['CW_test_acc'], csv_row['CW_test_loss'], _, _ = trainer.Linf_PGD_40(test_dataloader, 'CW')
    print('\n**Getting metrics on training data**')
    csv_row['PGD40_train_acc'], csv_row['PGD40_train_loss'], csv_row['clean_train_acc'], csv_row['clean_train_loss'] = trainer.Linf_PGD_40(train_dataloader, 'CE')
    csv_row['CW_train_acc'], csv_row['CW_train_loss'], _, _ = trainer.Linf_PGD_40(train_dataloader, 'CW')
    if epoch:
        csv_row['epoch'] = epoch
    if sampler_path:
        trainer.load_sampler_state(train_dataloader, sampler_path) 

def write_result_row(file_name, csv_row):
    row = pd.DataFrame(csv_row, index=[0])
    if os.path.exists(file_name):
        metrics = pd.read_csv(file_name)
        metrics = pd.concat([metrics, row], ignore_index=True)
        metrics.to_csv(file_name, index=False)
    else:
        row.to_csv(file_name, index=False)

def main():
    parser = argparse.ArgumentParser(description='Standard + Adversarial Training.')
    parser = parser_train(parser)
    parser = set_extra_arguments(parser)
    args = parser.parse_args()
    args = set_config_file_precedence(args)
    validate_train_arguments(args, parser)

    distributed_train(args)

if __name__ == '__main__':
    main()
