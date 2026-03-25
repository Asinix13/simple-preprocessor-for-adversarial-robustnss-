import kornia.filters as kf
import kornia.augmentation as ka
import kornia.enhance as ke
import torch
import torchattacks as ta
from torch import nn
import time as t
from kornia.core import Module, Tensor, pad, Device, Dtype, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
import kornia as k
from typing import Any, Optional, Union
from collections import OrderedDict

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

class Bil_layer(nn.Module):
    def __init__(self, bil = 0, kernel_size = 5, sigma_color=0.1, color2 = 0.01, sigma_space=10., blur = 0, noise = 0.0, median = 0, sup = 0, JPEG = 0, device = "cuda", q = 75):
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

        #for my bil_filter implementation
        space_kernel0 = get_gaussian_kernel2d(self.kernel_size, (self.sigma_space, self.sigma_space), device=self.device, dtype = torch.float)

        space_kernel1 = get_gaussian_kernel2d(3, (self.sigma_space, self.sigma_space), device=self.device, dtype = torch.float)
        
        # Register as non-trainable buffers
        self.register_buffer('space_Tensor0', space_kernel0, persistent=False)
        self.register_buffer('space_Tensor1', space_kernel1, persistent=False)


        print("bil=", self.bil, " median=", self.median, " blur=", self.blur, " noise=", self.noise, " color2=", self.color2, " sup=", self.sup, " jpeg=" , self.JPEG, "q=", self.q)

            
    def forward(self,x):

      
        if(self.noise!=0):
            n = torch.randn_like(x) 
            n *= self.noise**0.5 
            x = x + n
            
        
        if(self.bil != 0):
            x = my_bil(x,self.kernel_size, self.sigma_color, self.space_Tensor0)
            for _ in range(self.bil-1):
                x = my_bil(x,3, self.color2, self.space_Tensor1)
            

        if(self.JPEG!=0):
            q = torch.ones(x.shape[0], device=self.device, requires_grad=True)*self.q
            x = ke.JPEGCodecDifferentiable()(x,q)
     

        return x
    
    def print(self):
        print("sigma colour=", self.sigma_color.item(), " sigma space=", self.sigma_space.item(), " color2=", self.color2 )

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Return an empty state dictionary to prevent loading any parameters
        return OrderedDict()

    def load_state_dict(self, state_dict, strict=True):
        # If any keys related to this module exist in the state_dict, remove them
        keys_to_remove = [k for k in state_dict.keys() if k.startswith(f'{self.__class__.__name__}.')]
        for k in keys_to_remove:
            del state_dict[k]
        
        # Do nothing with the state dictionary
        return


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

            adv = adv + self.LEARNING_RATE * grad_sign

            adv = torch.clamp(adv, lower, upper)
        return adv

#test version
def wEoTBPDA(model, device, test_loader, epsilon):
    preprocessor = model.module[0]
    classifier = model.module[1]
    if(epsilon == 0):
        correct = 0
        for data, target in test_loader:
            data,target = data.to(device), target.to(device)

            data = preprocessor(data)

            output = classifier(data)
            # Check for success
            final_pred = output.max(1)[1] # get the index of the max log-probability
            final_true_mat = final_pred == target

            correct += (final_true_mat[final_true_mat == True]).size(dim=0)



    else:
        att = ta.APGD(classifier, eps=epsilon)
      
        # Accuracy counter
        correct = 0

        for data, target in test_loader:
            data,target = data.to(device), target.to(device)
            lower = torch.clamp(data - epsilon, 0, 1)
            upper = torch.clamp(data + epsilon, 0, 1)
            predata = preprocessor(data)

            perturbed_data = att(predata,target).to(device)

            # Re-classify the perturbed image
            output = classifier(preprocessor(torch.clamp(perturbed_data, lower, upper)))

            # Check for success
            final_pred = output.max(1)[1] # get the index of the max log-probability
            final_true_mat = final_pred == target

            correct += (final_true_mat[final_true_mat == True]).size(dim=0)
          
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader.dataset))  
    print(f"Epsilon: {epsilon}\tTest Accuracy  = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc

def EoTBPDA(model, device, test_loader, epsilon):
   
    if(len(model.module)) == 2:
        preprocessor = model.module[0]
        classifier = model.module[1]
        #size of the first bilateral filter kernel, of the gaussian kernel and of all median filtering kernels(following bilateral filters always uses 3)
        kernel_size = 5

        #the amount of bilateral filtering during inference; 0 is no bilateral filter 
        bil = 50

        #the sigma space for all bilateral filters
        sigma_spatial = 10.

        #the sigma range for the first bilateral filter
        sigma_color = .5

        #the sigma range for all following bilateral filters
        color2 = 0.05
    
        #the variance of the inherent Gaussian noise; 0 is no noise
        noise = 0.0

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
        EoTpreprocessor = Bil_layer(kernel_size=kernel_size, bil = preprocessor.bil, sigma_color=sigma_color, color2 = color2, sigma_space=sigma_spatial, blur = blur, noise = noise, median=med, sup = sup, JPEG=Jpeg,  q = q)
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

def my_bil(
    input: Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | Tensor,
    space_Tensor: Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1", 
) -> Tensor:
    

    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    
    if isinstance(sigma_color, Tensor):
        KORNIA_CHECK_SHAPE(sigma_color, ["B"])
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_input - input.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only accepts l1 or l2")
    
    
    color_kernel = ((-0.5 / sigma_color**2) * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    #0,32
    
    space_kernel = space_Tensor.view(-1, 1, 1, 1, kx * ky)
    
    #0,038
    
    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    

    return out

def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        KORNIA_CHECK(len(kernel_size) == 2, "2D Kernel size should have a length of 2.")
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)

def _compute_zero_padding(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2

def get_gaussian_kernel2d(
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | Tensor,
    force_even: bool = False,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the y and x direction. Sizes should be odd and positive.
        sigma: gaussian standard deviation in the y and x.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(B, \text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((5, 5), (1.5, 1.5))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                 [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                 [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]]])
        >>> get_gaussian_kernel2d((5, 5), torch.tensor([[1.5, 1.5]]))
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
    """
    if isinstance(sigma, tuple):
        sigma = tensor([sigma], device=device, dtype=dtype)

 

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "2"])

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]


    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device=device, dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device=device, dtype=dtype)[..., None]
    
    

    temp = kernel_y * kernel_x.view(-1, 1, ksize_x)
 

    return temp

def get_gaussian_kernel1d(
    kernel_size: int,
    sigma: float | Tensor,
    force_even: bool = False,
    *,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.

    Returns:
        gaussian filter coefficients with shape :math:`(B, \text{kernel_size})`.

    Examples:
        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([[0.3243, 0.3513, 0.3243]])
        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201]])
        >>> get_gaussian_kernel1d(5, torch.tensor([[1.5], [0.7]]))
        tensor([[0.1201, 0.2339, 0.2921, 0.2339, 0.1201],
                [0.0096, 0.2054, 0.5699, 0.2054, 0.0096]])
    """
    _check_kernel_size(kernel_size, allow_even=force_even)

    return gaussian(kernel_size, sigma, device=device, dtype=dtype)

def _check_kernel_size(kernel_size: tuple[int, ...] | int, min_value: int = 0, allow_even: bool = False) -> None:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)

    fmt = "even or odd" if allow_even else "odd"
    for size in kernel_size:
        KORNIA_CHECK(
            isinstance(size, int) and (((size % 2 == 1) or allow_even) and size > min_value),
            f"Kernel size must be an {fmt} integer bigger than {min_value}. Gotcha {size} on {kernel_size}",
        )

def gaussian(
    window_size: int,
    sigma: Tensor | float,
    *,
    mean: Optional[Union[Tensor, float]] = None,
    device: Optional[Device] = None,
    dtype: Optional[Dtype] = None,
) -> Tensor:
    """Compute the gaussian values based on the window and sigma values.

    Args:
        window_size: the size which drives the filter amount.
        sigma: gaussian standard deviation. If a tensor, should be in a shape :math:`(B, 1)`
        mean: Mean of the Gaussian function (center). If not provided, it defaults to window_size // 2.
        If a tensor, should be in a shape :math:`(B, 1)`
        device: This value will be used if sigma is a float. Device desired to compute.
        dtype: This value will be used if sigma is a float. Dtype desired for compute.
    Returns:
        A tensor withshape :math:`(B, \text{kernel_size})`, with Gaussian values.
    """

    if isinstance(sigma, float):
        sigma = tensor([[sigma]], device=device, dtype=dtype)

    KORNIA_CHECK_IS_TENSOR(sigma)
    KORNIA_CHECK_SHAPE(sigma, ["B", "1"])
    batch_size = sigma.shape[0]

    mean = float(window_size // 2) if mean is None else mean
    if isinstance(mean, float):
        mean = tensor([[mean]], device=sigma.device, dtype=sigma.dtype)

    KORNIA_CHECK_IS_TENSOR(mean)
    KORNIA_CHECK_SHAPE(mean, ["B", "1"])

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - mean).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)
