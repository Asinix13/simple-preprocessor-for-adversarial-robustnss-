import torch
from torchvision import models, transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd


import os
import datetime

import torchattacks as ta
from my_utils import bil_CIFAR10, Bil_layer

class transform_layer(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform(x)

#adaptad from https://github.com/Annonymous-repos/attacks-in-pytorch/blob/master/attacks/BPDA.py
class BPDAattack(object):
    def __init__(self, model=None, defense=None, device=None, epsilon=None, learning_rate=0.5,
                 max_iterations=50, clip_min=0, clip_max=1):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.defense = defense
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.device = device

    def generate(self, x, y):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.

        """

        adv = x.detach().clone()

        lower = torch.clamp(x - self.epsilon, self.clip_min, self.clip_max)
        upper = torch.clamp(x + self.epsilon, self.clip_min, self.clip_max)

        for i in range(self.MAX_ITERATIONS):
            adv_purified = self.defense(adv)
            adv_purified.requires_grad_()
            adv_purified.retain_grad()

            scores = self.model(adv_purified)
            loss = self.loss_fn(scores, y)
            loss.backward(retain_graph=True)

            grad_sign = adv_purified.grad.data.sign()

            # early stop, only for batch_size = 1
            # p = torch.argmax(F.softmax(scores), 1)
            # if y != p:
            #     break

            adv = adv + self.LEARNING_RATE * grad_sign

            adv = torch.clamp(adv, lower, upper)
        return adv

def multi_test(model, device, test_loader, epsilon, bil_layer):
    if(epsilon == 0):
        correct = [0,0,0,0,0]

        for data, target in test_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            # Check for success
            final_pred = output.max(1)[1] # get the index of the max log-probability
            final_true_mat = final_pred == target

            correct = [(x+sum(final_true_mat[final_true_mat == True])).item() for x in correct]

    else:
        att0 = ta.FGSM(model, eps=epsilon)
        att1 = ta.APGD(model, eps=epsilon)
        att2 = ta.EOTPGD(model, eps=epsilon)
        att3 = ta.APGD(model,norm='L2', eps=epsilon*17)
        Bpda = BPDAattack(model = model, defense= bil_layer, device=device, epsilon=epsilon, learning_rate = 2/255)

        att_l = [att0, att1, att2, att3]

        # Accuracy counter
        correct = [0,0,0,0,0]

        for data, target in test_loader:
            data,target = data.to(device), target.to(device)
            for i,att in enumerate(att_l):
                perturbed_data = att(data,target)
                # Re-classify the perturbed image
                output = model(perturbed_data)

                # Check for success
                final_pred = output.max(1)[1] # get the index of the max log-probability
                final_true_mat = final_pred == target

                correct[i] += (final_true_mat[final_true_mat == True]).size(dim=0)
            
            perturbed_data = Bpda.generate(data,target)
           
            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            final_pred = output.max(1)[1] # get the index of the max log-probability
            final_true_mat = final_pred == target

            correct[-1] += (final_true_mat[final_true_mat == True]).size(dim=0)

    # Calculate final accuracy for this epsilon
    final_acc = [x/float(len(test_loader.dataset)) for x in correct]
    print(f"Epsilon: {epsilon}\tTest Accuracy  = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc
           
now = datetime.datetime.now()
real_time = str(now.time())[0:8]

########################################### Hyperparameters ###########################################
batch_size=128

#where we save the data
path = str("./mult_adv/" + real_time)

#if a preprocessor layer is added in front of the NN
preproc = True

#the bilateral filter amount used for the training of the model
bil_m = 10

#the time and directory of the model
time ="14:44:15"
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

#if the data should be normalized:
transform = False

#the epsilons used for all attack, for the APGD L2 we multiply them by 17
epsilons = np.array([0, .05, .1, .15, .2, .3, 0.5, 1])/10

############################################################################################################

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


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


bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, sigma_space=sigma_spatial, color2= color2, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q).to(device)
if(preproc):    
    model = torch.nn.Sequential(bil_layer,model)

model_name = str(bil_m)+"_"+time
PATH= model_dir+"model_" + model_name
model.load_state_dict(torch.load(PATH))
if(test_transform != None):
    model = torch.nn.Sequential(transform_layer(test_transform).to(device),model)
model.eval()

epsilons = np.array([0, .05, .1, .15, .2, .3, 0.5, 1])/10
accuracies = []
examples = []

print("model name =", model_name,)

# Run test for each epsilon

for eps in epsilons:
    acc = multi_test(model, device, test_loader, eps, bil_layer)
    accuracies.append(acc)

#where we save the model
path = str("./mult_adv/" + real_time)

os.mkdir(path)

acc_df = pd.DataFrame(data=accuracies, index=epsilons).T
acc_df.to_excel(excel_writer = path +"/acc.xlsx")
   

with open(path + "/params.txt", "w") as f:
    print("bil=", bil, " median=", med, " blur=", blur, " noise=", noise, " color2=", color2," time=",time, " sup=", sup, "JPEG=", Jpeg, " q=", q,  file = f)