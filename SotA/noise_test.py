import os
import torch
import torch.nn as nn
import argparse
import logging
from utils_sparse import *
from data import load_cifar10
import kornia.augmentation as ka
from my_utils import Bil_layer
from dm_wide_resnet import MeanSparse_DMWideResNet
from MeanSparse_robustarch_wide_resnet import MeanSparse


def main():
    torch.cuda.empty_cache()
    print('GPU type:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    ########################### Initilization ############################
    parser = argparse.ArgumentParser(description="Applying Auto Attack")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--exp-name", type=str, default="noise")
    parser.add_argument("--start_batch", default=0, type=int)
    parser.add_argument("--end_batch", default=100, type=int)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--ckpt", type=str, help="checkpoint path for pretrained classifier")
    parser.add_argument("--print-freq", type=int, default=1)
    args = parser.parse_args()

    ########################################### Hyperparameters ###########################################
    #size of the first bilateral filter kernel, of the gaussian kernel and of all median filtering kernels(following bilateral filters always uses 3)
    kernel_size = 5

    #the amount of bilateral filtering during inference; 0 is no bilateral filter 
    bil = 10

    #the sigma space for all bilateral filters
    sigma_spatial = 10.

    #the sigma range for the first bilateral filter
    sigma_color = .5

    #the sigma range for all following bilateral filters
    color2 = 0.05

    #the variance of the inherent Gaussian noise; 0 is no noise
    noise = 0.032

    #the sigma space for Gaussian blurring; 0 is no Gaussian blurring
    blur = 0.

    #the amount of median filtering, 0 is no median filtering
    med = 0

    #the propability of inherant S&P noise; 0 is no noise
    sup = 0.

    #must be 0 or batch size; 0 is no jpeg transformation
    Jpeg = 0

    #quality of the jpeg image; between 100 and 0
    q = 0

    #variance of the gaussian noise only one variance at the time can be tested
    var = 0.00
    ############################################################################################################

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    result_sub_dir = os.path.join(args.results_dir, args.exp_name)
    create_subdirs(result_sub_dir)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "Results.log"), "a")
    )
    logger.info(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    ########################### Creating model ############################
    name_model, threshold ='Bartoldson2024Adversarial_WRN-94-16', 0.15 #Rank: New submission
    directory_WS = "models_WS"

    if not os.path.exists(directory_WS):
        os.makedirs(directory_WS, exist_ok=True)

    model = MeanSparse_DMWideResNet(num_classes=10,
                depth=94,
                width=16,
                activation_fn=nn.SiLU,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2471, 0.2435, 0.2616))
    
    bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, color2 = color2, sigma_space=sigma_spatial, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q)
    model = torch.nn.Sequential(bil_layer,model)

    file_path = os.path.join(directory_WS, '%s_WS.pt'%name_model)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    ########################### Loading Data ############################
    x_test, y_test = load_cifar10(n_examples=10000)

    gauss_noise = ka.RandomGaussianNoise(mean=0, std= var**0.5 , p =1)

    x_test = gauss_noise(x_test)

    print('Dataset Size:', len(y_test))
    
    ######################################
    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.threshold.data.fill_(threshold)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    ########################### Calculate accuracy ############################
    
    
    results = calculate_accuracy(model, x_test.to(device), y_test.to(device), batch_size=args.batch_size)
    logger.info('MeanSparse Model Clean Accuracy:', results)


if __name__ == "__main__":
    main()