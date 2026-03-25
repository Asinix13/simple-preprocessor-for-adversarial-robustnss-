#This file is adapted from https://github.com/locuslab/fast_adversarial
#from the "Fast is better then free: Revisiting adversarial training" paper





import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from torchvision import  models
import datetime

from utils import (clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)

def main():
    now = datetime.datetime.now()
    time0 = str(now.time())[0:8]

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)

########################################### Hyperparameters ###########################################
    batch_size = 64
    epochs = 80

    #model name and where we save the model
    model_name = "0_"+str(time0)
    path_to_model = "./Models/model_" + model_name

    # fast is better then free hyperparams
    epsilon = (8 / 255.) / std
    alpha = (10 / 255.) / std
    pgd_alpha = (2 / 255.) / std

    delta_init = 'random'
    lr_schedule = 'cyclic'
    early_stop = True
    lr_min = 0
    lr_max = 0.05
    loss_scale = '1.0'
############################################################################################################
    
    opt_level = "O2"
    master_weights = True
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device", f"; Using {opt_level} optimization level")

    #datasets init:

    model = models.efficientnet_b0().cuda()
    model.train()
    train_loader, test_loader = get_loaders(".", batch_size)

    opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=0.99, weight_decay=5e-4)
    amp_args = dict(opt_level=opt_level, loss_scale=loss_scale, verbosity=False)
    if opt_level == 'O2':
        amp_args['master_weights'] = master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    if delta_init == 'previous':
        delta = torch.zeros(batch_size, 3, 32, 32).cuda()

    lr_steps = epochs * len(train_loader)
    if lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            
            scheduler.step()
        if early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
    if not early_stop:
        best_state_dict = model.state_dict()

    
    torch.save(best_state_dict, path_to_model)
    

    # Evaluation
    model_test = models.efficientnet_b0().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 1, 1)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    print(test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()