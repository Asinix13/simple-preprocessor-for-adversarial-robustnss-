import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
import datetime
import torchattacks as ta
from my_utils import bil_CIFAR10, Bil_layer


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

    # Calculate final accuracy for this epsilon
    f_l2 = [x/float(len(test_loader.dataset)) for x in l2]
    final_acc = [x/float(len(test_loader.dataset)) for x in correct]
    print(f"Epsilon: {1}\tTest Accuracy  = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, f_l2
           
now = datetime.datetime.now()
real_time = str(now.time())[0:8]

########################################### Hyperparameters ###########################################
batch_size=128

#where we save the data
path = str("./CW_adv/" + real_time)

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

cifartest = bil_CIFAR10(".",  noise=0, transform=test_transform, download=True, device=device, train=False)
test_loader = DataLoader(cifartest, batch_size=batch_size, shuffle=True)

model = models.efficientnet_b0().to(device)


bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, sigma_space=sigma_spatial, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q).to(device)
if(preproc):    
    model = torch.nn.Sequential(bil_layer,model)
    
accuracies = []
l2s = []

print("model name =", time_list)

#check if all the model names are correct
for time in time_list:
    model_name = str(bil_m)+"_"+time
    PATH= model_dir + "model_" + model_name
    print(PATH)
    model.load_state_dict(torch.load(PATH))

# Run test for each epsilon
for time in time_list:
    model_name = str(bil_m)+"_"+time
    PATH=model_dir +"model_" + model_name
    model.load_state_dict(torch.load(PATH))
    model.eval()
    acc,l2 = test(model, device, test_loader)
    accuracies.append(acc)
    l2s.append(l2)

path = str("./CW_adv/" + real_time)

os.mkdir(path)

acc_df = pd.DataFrame(data=accuracies).T
acc_df.to_excel(excel_writer = path +"/acc.xlsx")

l2_df = pd.DataFrame(data=l2s).T
l2_df.to_excel(excel_writer = path +"/l2.xlsx")
   

with open(path + "/params.txt", "w") as f:
    print("bil=", bil, " median=", med,  " blur=", blur, " noise=", noise, " color2=", color2," time=",time, " sup=", sup, "JPEG=", Jpeg, " q=", q,  file = f)
