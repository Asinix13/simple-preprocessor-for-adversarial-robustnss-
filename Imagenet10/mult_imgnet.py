import torch
from torchvision import models, transforms
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from imagenet_dataloader import get_imagenet10_loader


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
    
class Identity_layer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

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

def multi_test(model, device, test_loader, epsilon):
    if(epsilon == 0):
        correct = [0,0,0,0]

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
      

        att_l = [att0,att1,att2]

        # Accuracy counter
        correct = [0,0,0,0]

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
          
    # Calculate final accuracy for this epsilon
    final_acc = [round(x/float(len(test_loader.dataset))*100, 2) for x in correct]
    print(f"Epsilon: {epsilon}\tTest Accuracy  = {(final_acc)}")

    # Return the accuracy and an adversarial example
    return final_acc

def EoTBPDA(model, device, test_loader, epsilon, bil_layer):
   
    if(False) == 2:
        preprocessor = model.module[0]
        classifier = model.module[1]
        #size of the first bilateral filter kernel, of the gaussian kernel and of all median filtering kernels(following bilateral filters always uses 3)
        kernel_size = 5

        #the amount of bilateral filtering during inference; 0 is no bilateral filter 
        bil = 20

        #the sigma space for all bilateral filters
        sigma_spatial = 50.

        #the sigma range for the first bilateral filter
        sigma_color = .2

        #the sigma range for all following bilateral filters
        color2 = 0.01
    
        #the variance of the inherent Gaussian noise; 0 is no noise
        noise = 0.008

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
        #EoTpreprocessor = Bil_layer(kernel_size=kernel_size, bil = preprocessor.bil, sigma_color=sigma_color, color2 = color2, sigma_space=sigma_spatial, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q)
        EoTpreprocessor = bil_layer
    else:
        preprocessor = Identity_layer()
        EoTpreprocessor = Identity_layer()
        classifier = model

    if(epsilon == 0):
        correct = 0
        for data, target in test_loader:
            data,target = data.to(device), target.to(device)

            #data = preprocessor(data)

            output = classifier(preprocessor(data))
            # Check for success
            final_pred = output.max(1)[1] # get the index of the max log-probability
            final_true_mat = final_pred == target

            correct += (final_true_mat[final_true_mat == True]).size(dim=0)



    else:
        att = BPDAattack(model=classifier, defense=EoTpreprocessor,device= device, epsilon= epsilon, max_iterations=10)
      
        # Accuracy counter
        correct = 0

        for data, target in test_loader:
            data,target = data.to(device), target.to(device)
                
           
            perturbed_data = att.generate(data, target)

            # Re-classify the perturbed image
            output = classifier(preprocessor(perturbed_data))

            # Check for success
            final_pred = output.max(1)[1] # get the index of the max log-probability
            final_true_mat = final_pred == target

            correct += (final_true_mat[final_true_mat == True]).size(dim=0)
          
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader.dataset))  
    print(f"Epsilon: {epsilon}\tTest Accuracy  = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc

def test(preproc: bool = True, time: str = "15:32:05", noise: float = 0.08, bil: int = 10, sigma_spatial: float = 0, sigma_color: float = 0, color2: float = 0):
    now = datetime.datetime.now()
    real_time = str(now.time())[0:8]

    ########################################### Hyperparameters ###########################################
    batch_size=16

    #where we save the data
    path = str("./mult_adv/" + real_time)

    #if a preprocessor layer is added in front of the NN
    preproc = preproc
    #the bilateral filter amount used for the training of the model
    bil_m = bil

    #the time and directory of the model
    time = time
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

    #if the data should be normalized:
    transform = False

    #the epsilons used for all attack, for the APGD L2 we multiply them by 17
    epsilons = np.array([0, .05, .1, .15, .2, .3, 0.5, 1])/20

    #epsilons = np.array([0, .05])/20

    EoTBPDA_flag = False

    ############################################################################################################

    test_transform = None

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using {device} device")

    root = '../imagenet10'
    data_dim = 256
    train_test_split = [0.8, 0.2]

    _, test_loader = get_imagenet10_loader(root, data_dim, train_test_split, batch_size)

    model = models.efficientnet_b0().to(device)

    model_name = str(bil_m)+"_"+time
    PATH= model_dir+"/Bil_" + model_name + "/" + "model"

    # Load the state dict
    state_dict = torch.load(PATH)

    # Remove 'module.' prefix from DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)



    if(preproc):
        bil_layer = Bil_layer(kernel_size=kernel_size, bil = bil, sigma_color=sigma_color, sigma_space=sigma_spatial, color2=color2, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q).to(device)
        model = torch.nn.Sequential(bil_layer,model)



    if(test_transform != None):
        model = torch.nn.Sequential(transform_layer(test_transform).to(device),model)
    model = torch.nn.DataParallel(model) 
    model.eval()


    accuracies = []
    examples = []

    print("model name =", model_name,)

    # Run test for each epsilon

    for eps in epsilons:
        if not EoTBPDA_flag:
            acc = multi_test(model, device, test_loader, eps)
            acc[3] = EoTBPDA(model, device, test_loader, eps, bil_layer)
        accuracies.append(acc)
    #where we save the model
    path = str(model_dir+"/Bil_" + model_name + "/")


    acc_df = pd.DataFrame(data=accuracies, index=epsilons)
    acc_df.to_csv(path + "/acc.csv", index=True, float_format="%.4f")
    
def main():
    noise_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    time_list = ["16:33:43", "17:48:03", "17:58:25", "19:10:24", "20:33:54"]
    sigma_color = [50, 10, 0.5, 0.01, 0.005] # 0.2
    color2 =[1, 0.5, 0.01, 0.0005, 0.0001] # 0.006
    

    for i in range(5):
        test(preproc = True, time = time_list[i], bil = 0, noise = 0 , sigma_spatial = 50, sigma_color = 10, color2= 0.5)
        
main()