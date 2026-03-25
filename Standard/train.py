
import numpy as np
import torch
from torchvision import  models, transforms
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import os
import datetime
from my_utils import bil_CIFAR10, Bil_layer


def run_ef03():
    #hyperparams

    #the amount of bilateral filter iterations, if bil = 0 then there is no bilateral filtering

    now = datetime.datetime.now()
    time = str(now.time())[0:8]
    
    ########################################## Hyperparameters ###########################################
    batch_size = 64
    epochs = 80

    #if a preprocessor layer is added in front of the NN
    preproc = False

    #where the model should be saved
    model_dir = "./Models/"

    result_dir = "./Results/"

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

    #if the classwise accuracys should be calculated
    classwiseacc = True

    #if the model should be tested
    Test = True

    #if gaussian noise shlould be used during testing
    gauss = True

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
    if(gauss):
        noises = [0, 0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.02, 0.03]

    cif_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    
    model_name = str(bil)+"_"+str(time)
    path_to_model = model_dir + "model_" + model_name
    path =result_dir + "Bil_" + str(bil)+"_"+ time
 
    print("noises=", noises, " gaussian=", gauss, " normalization=", transform )

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
    

    bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, sigma_space=sigma_spatial, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q, device=device).to(device)
    if(preproc):    
        model = torch.nn.Sequential(bil_layer,model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    cifartrain = bil_CIFAR10(".",  noise=0, download=True, transform=test_transform, device=device)
    train_dataloader = DataLoader(cifartrain, batch_size=batch_size, shuffle=True)

    cifartests = [bil_CIFAR10(".", noise=n, train=False, download=True, transform=test_transform) for n in noises]
    test_dataloaders = [DataLoader(d, batch_size=batch_size) for d in cifartests]
    

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            y =  y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            if(classwiseacc):
                    cacc = torch.zeros(10,device=device)
                    frog = torch.zeros(10,device=device)
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                temp = (pred.argmax(1) == y).type(torch.float)
                correct += temp.sum().item()
                if(classwiseacc):
                    cacc.index_add_(0,y,temp)
                    temp = (pred.argmax(1) == 6).type(torch.float)
                    frog.index_add_(0,y,temp)

        if(classwiseacc):         
            cacc *= (10/size)*100
            frog *= (10/size)*100    
        test_loss /= num_batches
        correct /= size
        
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        return correct*100, cacc.cpu(), frog.cpu()
    
    var = np.zeros((len(noises),10))
    c_var = np.zeros((len(noises),10,10))
    frog = np.zeros((len(noises),10,10))
    
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)  
        i = 0
        if(Test):
            for test_dataloader in test_dataloaders:
                if(t >= epochs-10):
                    var[i,t-epochs], c_var[i,t-epochs], frog[i,t-epochs] = test(test_dataloader, model, loss_fn)
                    i+=1


    if(Test):
        mean, me_var = np.mean(var,-1), np.std(var,-1)
        print(mean, me_var)

        os.mkdir(path)

        m_df = pd.DataFrame(data=mean, index=noises).T
        m_df.to_excel(excel_writer = path +"/m.xlsx")

        v_df = pd.DataFrame(data=me_var, index=noises).T
        v_df.to_excel(excel_writer = path +"/v.xlsx")

    if(classwiseacc):
        d = np.transpose(np.mean(c_var[:,:,:],-2))
        m_d = pd.DataFrame(data=d, index=cif_dict.values(), columns=noises)
        m_d.to_excel(excel_writer = path +"/full_m.xlsx")
    
        d = np.transpose(np.std(c_var[:,:,:],-2))
        m_d = pd.DataFrame(data=d, index=cif_dict.values(), columns=noises)
        m_d.to_excel(excel_writer = path +"/full_v.xlsx")


    print("bil=", bil, " median=", med, " gaussian=", gauss,  " blur=", blur, " noise=", noise)
    print("Noise:", noises)
    bil_layer.print()

    torch.save(model.state_dict(), path_to_model)
    with open(path + "/params.txt", "w") as f:
        print("bil=", bil, " median=", med, " blur=", blur, " noise=", noise, " color2=", color2," time=",time, " sup=", sup, "JPEG=", Jpeg, " q=", q,  file = f)


run_ef03()

