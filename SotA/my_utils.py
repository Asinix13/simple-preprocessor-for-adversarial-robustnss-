import kornia.filters as kf
import kornia.augmentation as ka
import kornia.enhance as ke
import torch
import torchattacks as ta
from torch import nn

class Bil_layer(nn.Module):
    def __init__(self, bil = 0, kernel_size = 3, sigma_color=0.1, color2 = 0.01, sigma_space=10, blur = 0, noise = 0.0, median = 0, sup = 0, JPEG = 0, device = "cuda", q = 75):
        super().__init__()
        self.bil = bil
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.noise = noise
        self.blur = (blur,blur)
        self.median = median
        self.sup = sup
        self.JPEG = JPEG
        #0.01 is standard
        self.color2 = color2
        self.device = device
        self.q = q
        

        print("bil=", self.bil, " median=", self.median, " blur=", self.blur, " noise=", self.noise, " color2=", self.color2, " sup=", self.sup, " jpeg=" , self.JPEG, "q=", self.q)

            
    def forward(self,x):
        if(self.noise):
            x = ka.RandomGaussianNoise(mean=0, std= self.noise**0.5 , p =1)(x)
        
        if(self.sup):
            x = ka.RandomSaltAndPepperNoise(self.sup, 0.5, 1)(x)
        
        if(self.blur[0]):
            x = kf.gaussian_blur2d(x, self.kernel_size, self.blur)

        if(self.bil):
            x = kf.bilateral_blur(x,self.kernel_size, self.sigma_color, (self.sigma_space,self.sigma_space))
            for _ in range(self.bil-1):
                x = kf.bilateral_blur(x,3, self.color2, (self.sigma_space,self.sigma_space))

        if(self.median):
            x = kf.median_blur(x, 3)         
            for i in range(self.median-1):
                x = kf.median_blur(x, 3)
        
        if(self.JPEG):
            q = torch.ones(x.shape[0], device=self.device, requires_grad=True)*self.q
            x = ke.JPEGCodecDifferentiable()(x,q)

        return x
    
    def print(self):
        print("sigma colour=", self.sigma_color.item(), " sigma space=", self.sigma_space.item(), " color2=", self.color2 )

def multi_test(model, device, test_loader, epsilon, bil_layer):
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
        att3 = ta.APGD(model,norm='L2', eps=epsilon*17)
      

        att_l = [att0,att1,att2,att3]

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
    final_acc = [x/float(len(test_loader.dataset)) for x in correct]
    print(f"Epsilon: {epsilon}\tTest Accuracy  = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc
           