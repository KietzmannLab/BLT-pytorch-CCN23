import argparse
parser = argparse.ArgumentParser(description='Obtaining hyps')
parser.add_argument('--timesteps', type=int, default=10)
parser.add_argument('--lateral_connections', type=int, default=1)
parser.add_argument('--topdown_connections', type=int, default=0)
parser.add_argument('--LT_interaction', type=str, default='additive')
parser.add_argument('--LT_position', type=str, default='all')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--load_epoch', type=int, default=-2)
parser.add_argument('--classifier_bias', type=int, default=0)
parser.add_argument('--norm_type', type=str, default='LN')
args = parser.parse_args()

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

from helpers.helper_funcs import get_Dataset_loaders, get_folders_logging
from models.helper_funcs import get_network_model, compute_accuracy, eval_network

##################
## Hyperparameters
##################

hyp = {
    'dataset': {
        'name': 'miniecoset', # name of the dataset
        'image_size': args.image_size, # determines which dataset to load
        'dataset_path': '/share/klab/datasets/', # Folder where dataset exists (end with '/')
        'augment': {'normalize'} # Mention augmentations to be used here - trivialaug, autoaugment, randaugment, normalize
    },
    'network': {
        'model': 'BLT_net', # model to be used
        'identifier': '1', # identifier in case we run multiple versions of the net
        'timesteps': args.timesteps, # number of timesteps to unroll the RNN
        'lateral_connections': args.lateral_connections, # if lateral connections should exist throughout the network
        'topdown_connections': args.topdown_connections, # if topdown connections should exist throughout the network
        'LT_interaction': args.LT_interaction, # 'additive' or 'multiplicative' interaction with bottom-up flow
        'LT_position': args.LT_position, # 'all' = everywhere, 'last' = at the GAP layer
        'classifier_bias': args.classifier_bias, # if the classifier layer should have a bias parameter
        'norm_type': args.norm_type # which norm to use - 'LN', 'None'
    },
    'optimizer': {
        'batch_size': args.batch_size, # batch_size for dataloader
        'device': 'cuda', # device to train the network on: 'cuda', 'mps', 'cpu'
        'dataloader': {
            'num_workers_train': 10, # number of cpu workers processing the batches 
            'prefetch_factor_train': 4, # number of batches kept in memory by each worker (providing quick access for the gpu)
            'num_workers_val_test': 3, # do not need lots of workers for val/test
            'prefetch_factor_val_test': 4 
        }
    },
    'misc': {
        'load_epoch': args.load_epoch, # either mention epoch name (positive int or -1 for random init) or -2 (which loads the final saved model)
    }
}

##########################
## Evaluation
##########################

if __name__ == '__main__':

    # load the test loader for eval - includes transform
    _, _, _, testplus_loader = get_Dataset_loaders(hyp)

    # create the network and load weights
    net, net_name = get_network_model(hyp)
    net = net.float()
    net.to(hyp['optimizer']['device'])
    print('Epoch: ',hyp['misc']['load_epoch'],'\n')

    log_path, net_path, save_actvs_path = get_folders_logging(net_name)
    print(f'Net_folder: {net_path}\n')

    if hyp['misc']['load_epoch'] == -2:
        net_save_path = f'{net_path}/{net_name}.pth'
    else:
        load_epoch = hyp['misc']['load_epoch']
        net_save_path = f'{net_path}/{net_name}_epoch_{load_epoch}.pth'
        
    net.load_state_dict(torch.load(net_save_path,map_location=torch.device(hyp['optimizer']['device'])))
    net.eval()

    print('Net loaded! Evaluation begins here!\n')

    outputs_all = np.zeros([hyp['network']['timesteps'],25000,100])
    labels_all = np.zeros([25000,])
    representations_all = np.zeros([hyp['network']['timesteps'],25000,512])

    torch.cuda.synchronize()

    count_h = 0

    for images,labels in testplus_loader:

        imgs = images.to(hyp['optimizer']['device'])
        lbls = labels.to(hyp['optimizer']['device'])
        
        outputs, representations = net(imgs)

        len_out = outputs[0].shape[0]
        
        for t in range(hyp['network']['timesteps']):
            outputs_all[t,count_h:count_h+len_out,:] = outputs[t].cpu().detach().numpy()
            representations_all[t,count_h:count_h+len_out,:] = representations[t].cpu().detach().numpy()
        
        labels_all[count_h:count_h+len_out] = lbls.cpu().detach().numpy()

        count_h += len_out

    test_acc_running = eval_network(testplus_loader,net,hyp)
    test_acc = test_acc_running/len(testplus_loader)
    print(f'Test accuracies over time (%): {test_acc}\n')

    readout_weight = net.readout.weight.cpu().detach().numpy()
    if args.classifier_bias:
        readout_bias = net.readout.bias.cpu().detach().numpy()
    else:
        readout_bias = []
    
    print('Evaluation data has been registered!\n')

    # import pdb; pdb.set_trace()
    
    if hyp['misc']['load_epoch'] == -2:
        save_act_path_name = f'{save_actvs_path}/{net_name}.npz'
    else:
        save_act_path_name = f'{save_actvs_path}/{net_name}_epoch_{load_epoch}.npz'

    np.savez(save_act_path_name, outputs_all=outputs_all, labels_all=labels_all, representations_all=representations_all,
             readout_weight=readout_weight, readout_bias=readout_bias)