
import numpy as np
import torch
from torchvision import  models, transforms
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import os
import datetime
from my_utils import bil_CIFAR10, Bil_layer
from imagenet_dataloader import get_imagenet10_loader
import time as ti


def run_ef03(preproc: bool = True, noise: float = 0.08, bil: int = 10, sigma_spatial: float = 0, sigma_color: float = 0, color2: float = 0):
    #hyperparams

    #the amount of bilateral filter iterations, if bil = 0 then there is no bilateral filtering

    now = datetime.datetime.now()
    time = str(now.time())[0:8]
    
    ########################################## Hyperparameters ###########################################
    batch_size = 128
    epochs = 60

    #if a preprocessor layer is added in front of the NN
    preproc = preproc

    #where the model should be saved
    model_dir = "./Models/"

    #size of the first bilateral filter kernel, of the gaussian kernel and of all median filtering kernels(following bilateral filters always uses 3)
    kernel_size = 5

    #the amount of bilateral filtering during inference; 0 is no bilateral filter 
    bil = bil

    #the sigma space for all bilateral filters
    sigma_spatial = sigma_spatial

    #the sigma range for the first bilateral filter
    sigma_color = sigma_color

    #the sigma range for all following bilateral filters
    color2 = color2

    #the variance of the inherent Gaussian noise; 0 is no noise
    noise = noise

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

    #if the model should be tested
    Test = True



    ############################################################################################################


    model_name = str(bil)+"_"+str(time)
    folder_path = model_dir + "Bil_" + model_name 
    path_to_model = folder_path + "/model"
 

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    #datasets init:
    efnetb0 = models.efficientnet_b0()
    model = efnetb0.to(device)

    loss_fn = nn.CrossEntropyLoss()
 
    
    if(preproc):    
        bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, sigma_space=sigma_spatial, color2=color2, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q).to(device)
        model = torch.nn.Sequential(bil_layer,model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    root = '../imagenet10'
    data_dim = 256
    train_test_split = [0.8, 0.2]
    
    train_dataloader, test_dataloader = get_imagenet10_loader(root, data_dim, train_test_split, batch_size)

    model = torch.nn.DataParallel(model)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 24 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    
 
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        start = ti.time()
        train(train_dataloader, model, loss_fn, optimizer)  
        end = ti.time()
        print(f"Time: {(end-start):>4f} \n")
        if(Test and (t >= epochs-10 or t <= 4)):
            test(test_dataloader, model, loss_fn)

    
    os.mkdir(folder_path)
    
    if(preproc):
        torch.save(model.module[1].state_dict(), path_to_model)
        print("bil=", bil, " median=", med,  " blur=", blur, " noise=", noise)
        bil_layer.print()
    else:
        torch.save(model.state_dict(), path_to_model)

    with open(folder_path + "/params.txt", "w") as f:
        print("prepro=", preproc," time=", time, "bil=", bil,  " noise=", noise, " color2=", color2, " sigma_spatial=", sigma_spatial, " sigma_color=", sigma_color, " swap preproc", file = f)

def main():
    noise_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    sigma_color = [50, 10, 0.5, 0.01, 0.005] # 0.2
    color2 =[1, 0.5, 0.01, 0.0005, 0.0001] # 0.006

    for i in range(5):
        run_ef03(preproc = True, bil = 0, noise = 0, sigma_spatial = 50, sigma_color = 10, color2= 0.5)

main()


