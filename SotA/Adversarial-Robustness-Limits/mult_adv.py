import os
import torch
import torch.nn as nn
import numpy as np
from data import load_cifar10
import pandas as pd
import datetime
import json
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader

from core.models.my_utils import multi_test
from core.models.my_utils import EoTBPDA
from core.models import create_model
from core.data import get_data_info
from core.utils import parser_eval, set_config_file_precedence, set_extra_arguments,validate_train_arguments



def main():
    torch.cuda.empty_cache()
    print('GPU type:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    ########################### Initilization ############################
    
    now = datetime.datetime.now()
    real_time = str(now.time())[0:8]

    parse = parser_eval()
    parse = set_extra_arguments(parse)
    args = parse.parse_args()
    args = set_config_file_precedence(args)

    LOG_DIR = args.log_dir + '/' + args.desc
    with open(LOG_DIR+'/args.txt', 'r') as f:
        old = json.load(f)
        args.__dict__ = dict(vars(args), **old)
    args.config = 'configs/test.json'
    validate_train_arguments(args, parse)
    ########################################### Hyperparameters ###########################################
    batch_size=200

    #where we save the data
    path = str("./mult_adv/" + real_time)
    ############################################################################################################
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    ########################### Creating model ############################
    directory_WS = "models_WS"

    data_dir = './mnt/MLdata/CIFAR-10-EDM'
    DATA_DIR = data_dir + '/cifar10/'

    WEIGHTS = LOG_DIR + '/weights-best.pt'

    print(WEIGHTS)

    info = get_data_info(DATA_DIR)

    if not os.path.exists(directory_WS):
        os.makedirs(directory_WS, exist_ok=True)

    model = create_model(args.model, args.normalize, info, device, args)
    checkpoint = torch.load(WEIGHTS)
    if 'tau' in args and args.tau:
        print ('Using WA model.')
    def distributed_safe_load(checkpoint):
        msd = deepcopy(checkpoint['model_state_dict'])
        if 'module'==list(checkpoint['model_state_dict'].keys())[0][:6]:
            for k in checkpoint['model_state_dict']:
                assert k[:7] == 'module.'
                msd[k[7:]] = msd[k]
                del msd[k]
        return msd
    model.load_state_dict(distributed_safe_load(checkpoint))
    model = torch.nn.DataParallel(model) 
    model.eval()
    del checkpoint


    ########################### Loading Data ############################
    x_test, y_test = load_cifar10(n_examples=10000, data_dir='./mnt/MLdata/cifar10')
    print('Dataset Size:', len(y_test))
    
   
    _dataset = TensorDataset(x_test[:], y_test[:])
    data_loader = DataLoader(_dataset, batch_size=batch_size, shuffle=None, sampler=None, pin_memory=True)
    
    ########################### Applying auto attack ############################
    epsilons = np.array([0, .05, .1, .15, .2, .3, 0.5, 1])/10
    accuracies = []

    # Run test for each epsilon
    if args.TABPDA:
        for eps in epsilons:
            acc = EoTBPDA(model, device, data_loader, eps)
            accuracies.append(acc)

    else:
        for eps in epsilons:
            acc = multi_test(model, device, data_loader, eps)
            accuracies.append(acc)


    os.mkdir(path)

    acc_df = pd.DataFrame(data=accuracies, index=epsilons).T
    acc_df.to_excel(excel_writer = path +"/acc.xlsx")
    

    with open(path + "/params.txt", "w") as f:
        print("model = ",args.model, "bil = ", args.bil, "gvar = ", args.gvar ,file = f)

if __name__ == "__main__":
    main()
