
import torch
from torchvision import  models, transforms
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import os
import datetime
import numpy as np

from my_utils import bil_CIFAR10, Bil_layer

def run_ef03():
    now = datetime.datetime.now()
    real_time = str(now.time())[0:8]

    ########################################### Hyperparameters ###########################################
    batch_size=128

    #where we save the data
    path = str("./Results/" + real_time)

    #if a preprocessor layer is added in front of the NN
    preproc = True

    #the bilateral filter amount used for the training of the model
    bil_m = 10

    #the times and directory of the models, here we can use multiple models at the same time
    time_list = ["14:44:15"]
    model_dir = "./Models/"

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

    # if the data should be normalized:
    transform = False

    ############################################################################################################
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    if(transform):
        test_transform = transforms.Compose([
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    else:
        test_transform = None

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using {device} device")

    noises = [0, 0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.02, 0.03]

    cifartests = [bil_CIFAR10(".", noise=n, train=False, download=True, transform=test_transform) for n in noises]
    test_dataloaders = [DataLoader(d, batch_size=batch_size) for d in cifartests]

    model = models.efficientnet_b0().to(device)

    bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, sigma_space=sigma_spatial, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q).to(device)
    if(preproc):    
        model = torch.nn.Sequential(bil_layer,model)

    loss_fn = nn.CrossEntropyLoss()

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                temp = (pred.argmax(1) == y).type(torch.float)
                correct += temp.sum().item()
                
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct*100
    

    print("model name =", time_list)

    #check if all the model names are correct
    for time in time_list:
        model_name = str(bil_m)+"_"+time
        PATH= model_dir + "model_" + model_name
        print(PATH)
        model.load_state_dict(torch.load(PATH))

    accuracies = []

    # Run test for each model
    for time in time_list:
        model_name = str(bil_m)+"_"+time
        PATH=model_dir +"model_" + model_name
        model.load_state_dict(torch.load(PATH))
        model.eval()
        acc = np.zeros((len(noises)))
        for i,test_dataloader in enumerate(test_dataloaders):
            acc[i]= test(test_dataloader, model, loss_fn)
        accuracies.append(acc)
    
    path = str("./Noise_test/" + real_time)

    os.mkdir(path)

    acc_df = pd.DataFrame(data= accuracies).T
    acc_df.to_excel(excel_writer = path +"/acc.xlsx")
    

    with open(path + "/params.txt", "w") as f:
        print("bil=", bil, " median=", med,   " blur=", blur, " noise=", noise, " color2=", color2," time=",time, " sup=", sup, "JPEG=", Jpeg, " q=", q,  file = f)

run_ef03()