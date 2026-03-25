from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import torchattacks as ta
import os
import torch
from utils_sparse import *
from dm_wide_resnet import MeanSparse_DMWideResNet
from MeanSparse_robustarch_wide_resnet import MeanSparse
import pandas as pd
import kornia.augmentation as ka

from my_utils import Bil_layer

from data import load_cifar10


def test( model, device, test_loader):
   
    correct = [0,0,0,0,0,0,0]
    
    for data, target in test_loader:
        data,target = data.to(device), target.to(device)
        output = model(data)
        # Check for success
        final_pred = output.max(1)[1] # get the index of the max log-probability
        final_true_mat = final_pred == target
        correct[-1] += (final_true_mat[final_true_mat == True]).size(dim=0)

    #the different C we are testing for   
    c = [1,1.2,1.4,1.6,1.8,2]

    att_l = [ta.CW(model = model,c = x) for x in c ]
    l2 = [0,0,0,0,0,0]

    # Loop over all examples in test set
    for data, target in test_loader:
            data,target = data.to(device), target.to(device)
            for i,att in enumerate(att_l):
                perturbed_data = att(data,target)
                # Re-classify the perturbed image
                output = model(perturbed_data)

                # Check for success
                final_pred = output.max(1)[1] # get the index of the max log-probability
                final_true_mat = final_pred == target

                l2[i] += torch.sum((data-perturbed_data)**2).item()
                correct[i] += (final_true_mat[final_true_mat == True]).size(dim=0)
                print(i)
    print(correct)
    # Calculate final accuracy for this epsilon
    f_l2 = [x/float(len(test_loader.dataset)) for x in l2]
    final_acc = [x/float(len(test_loader.dataset)) for x in correct]
    print(f"Epsilon: {1}\tTest Accuracy  = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, f_l2
           
def main():
    torch.cuda.empty_cache()
    print('GPU type:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())

    now = datetime.datetime.now()
    real_time = str(now.time())[0:8]
    ########################################### Hyperparameters ###########################################
    batch_size=12

    #where we save the data
    path = str("./CW_adv/" + real_time)

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
    ############################################################################################################
    

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

    file_path = os.path.join(directory_WS, '%s_WS.pt'%name_model)
    checkpoint = torch.load(file_path)

    bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, color2 = color2, sigma_space=sigma_spatial, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q)
    model = torch.nn.Sequential(bil_layer,model)

    model.load_state_dict(checkpoint)
    model.to(device)
    ########################### Loading Data ############################
    x_test, y_test = load_cifar10(n_examples=10000)
    print('Dataset Size:', len(y_test))

    _dataset = TensorDataset(x_test[:], y_test[:])
    data_loader = DataLoader(_dataset, batch_size=batch_size, shuffle=None, sampler=None, pin_memory=True)
    
    # switch to evaluation mode
    model.eval()
    
    ######################################
    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.threshold.data.fill_(threshold)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    ########################### Applying auto attack ############################
    
    accuracies = []
    l2s = []


    model.eval()
    acc,l2 = test(model, device, data_loader)
    accuracies.append(acc)
    l2s.append(l2)

    os.mkdir(path)

    acc_df = pd.DataFrame(data=accuracies).T
    acc_df.to_excel(excel_writer = path +"/acc.xlsx")

    l2_df = pd.DataFrame(data=l2s).T
    l2_df.to_excel(excel_writer = path +"/l2.xlsx")
    

    with open(path + "/params.txt", "w") as f:
        print("bil=", bil, " median=", med, " blur=", blur, " noise=", noise, " color2=", color2," time=",time, " sup=", sup, "JPEG=", Jpeg, " q=", q,  file = f)

if __name__ == "__main__":
    main()