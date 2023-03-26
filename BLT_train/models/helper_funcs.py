import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import h5py
import time
import os
    
##################################
## Importing the network
##################################

def get_network_model(hyp):
    # import the req. network

    if hyp['network']['model'] == 'BLT_net':

        l_flag = hyp['network']['lateral_connections']
        t_flag = hyp['network']['topdown_connections']
        ltinteract = hyp['network']['LT_interaction']
        T_info = hyp['network']['timesteps']
        ltpos = hyp['network']['LT_position']
        netnum = hyp['network']['identifier']
        classifier_bias = hyp['network']['classifier_bias']
        norm_type = hyp['network']['norm_type']
        if T_info == 1: # if only 1 timestep requested, then no LT
            l_flag = 0
            t_flag = 0
            ltinteract = None
            ltpos = None

        if hyp['dataset']['image_size'] == 64:

            from .BLT_net import BLT_net_64

            net = BLT_net_64(lateral_connections = l_flag, topdown_connections = t_flag, LT_interaction = ltinteract, timesteps = T_info,
                             LT_position = ltpos, classifier_bias = classifier_bias, norm_type = norm_type)
            net_name = f'b64_l_{l_flag}_t_{t_flag}_ltinteract_{ltinteract}_T_{T_info}_ltposition_{ltpos}_num_{netnum}'
            print(f'\nNetwork name: {net_name}')

        elif hyp['dataset']['image_size'] == 128:

            from .BLT_net import BLT_net_128

            net = BLT_net_128(lateral_connections = l_flag, topdown_connections = t_flag, LT_interaction = ltinteract, timesteps = T_info,
                             LT_position = ltpos, classifier_bias = classifier_bias, norm_type = norm_type)
            net_name = f'b128_l_{l_flag}_t_{t_flag}_ltinteract_{ltinteract}_T_{T_info}_ltposition_{ltpos}_classifier_bias_{classifier_bias}_norm_{norm_type}_num_{netnum}'
            print(f'\nNetwork name: {net_name}')        
    
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"\nThe network has {params} trainable parameters\n")

    return net, net_name

def weights_init(m):
    # Xavier intialisation
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def get_optimizer(hyp,net):
    # selecting the optimizer

    if hyp['optimizer']['type'] == 'adam':
        return optim.Adam(net.parameters(),lr=hyp['optimizer']['lr'])
    
def compute_accuracy(outputs,labels):
    
    timesteps = len(outputs)
    accuracies = [0 for t in range(timesteps)]
    
    for t in range(timesteps):
        _, predicted = torch.max(outputs[t].data, 1)
        total = labels.shape[0]
        correct = (predicted == labels).sum().item()
        accuracies[t] = correct*100./total
    
    return accuracies 
    
def eval_network(data_loader,net,criterion,hyp): 
    # during training, evaluate network on val or test images (val_loader or test_loader)

    with torch.no_grad():
        loss_running = 0.0
        acc_running = np.zeros([hyp['network']['timesteps'],])
        for images,labels in data_loader:
            imgs = images.to(hyp['optimizer']['device'])
            lbls = labels.to(hyp['optimizer']['device'])
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                loss = criterion(outputs[0], lbls.long())
                acc_running[0] += compute_accuracy([outputs[0]],lbls)[0]
                if hyp['network']['timesteps'] > 1:
                    for t in range(hyp['network']['timesteps']-1):
                        loss = loss + criterion(outputs[t+1], lbls.long())
                        acc_running[t+1] += compute_accuracy([outputs[t+1]],lbls)[0]
            loss_running += loss.item()
            loss_running = loss_running

    return loss_running, acc_running
