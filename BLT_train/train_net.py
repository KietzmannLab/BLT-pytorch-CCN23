import argparse
parser = argparse.ArgumentParser(description='Obtaining hyps')
parser.add_argument('--timesteps', type=int, default=10)
parser.add_argument('--lateral_connections', type=int, default=1)
parser.add_argument('--topdown_connections', type=int, default=0)
parser.add_argument('--LT_interaction', type=str, default='additive')
parser.add_argument('--LT_position', type=str, default='all')
parser.add_argument('--show_progress_bar', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--batch_size_val_test', type=int, default=250)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--classifier_bias', type=int, default=0)
parser.add_argument('--norm_type', type=str, default='LN')
parser.add_argument('--n_epochs', type=int, default=60)
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

from helpers.helper_funcs import get_Dataset_loaders, create_folders_logging
from models.helper_funcs import get_network_model, weights_init, get_optimizer, eval_network, compute_accuracy

##################
## Hyperparameters
##################

show_progress_bar = args.show_progress_bar # if you want to print the training progress bar

hyp = {
    'dataset': {
        'name': 'miniecoset', # name of the dataset
        'image_size': args.image_size, # determines which dataset to load
        'dataset_path': '/share/klab/datasets/', # Folder where dataset exists (end with '/')
        'augment': {'trivialaug','normalize'} # Mention augmentations to be used here - trivialaug, autoaugment, randaugment, normalize
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
        'type': 'adam', # optimizer to be used
        'lr': 0.001, # learning rate
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs, # number of epochs (full cycle through the dataset)
        'device': 'cuda', # device to train the network on: 'cuda', 'mps', 'cpu'
        'dataloader': {
            'num_workers_train': 10, # number of cpu workers processing the batches 
            'prefetch_factor_train': 4, # number of batches kept in memory by each worker (providing quick access for the gpu)
            'num_workers_val_test': 3, # do not need lots of workers for val/test
            'prefetch_factor_val_test': 4 
        }
    },
    'misc': {
        'use_amp': True, # use automatic mixed precision during training - forward pass .half(), backward full
        'batch_size_val_test': args.batch_size_val_test, # issue: {a single A100 could not handle 1k batch size for val}
        'save_logs': 10, # after how many epochs should we save a copy of the logs
        'save_net': 30 # after how many epochs should we save a copy of the net
    }
}

##########################
## Training and evaluation
##########################

if __name__ == '__main__':

    # load the dataset loaders to iterate over for training and eval (CS MAGIC)
    train_loader, val_loader, test_loader, hyp = get_Dataset_loaders(hyp)

    # create the network and initialize it
    net, net_name = get_network_model(hyp)
    net.apply(weights_init)
    net = net.float()
    net.to(hyp['optimizer']['device'])

    # criterion and optimizer setup
    criterion = nn.NLLLoss()
    optimizer = get_optimizer(hyp,net)
    scaler = torch.cuda.amp.GradScaler(enabled=hyp['misc']['use_amp']) # this is in service of mixed precision training

    # logging losses and accuracies
    train_losses = [0 for epoch in range(hyp['optimizer']['n_epochs'])]
    train_accuracies = [0 for epoch in range(hyp['optimizer']['n_epochs'])]
    val_losses = [0 for epoch in range(hyp['optimizer']['n_epochs'])]
    val_accuracies = [0 for epoch in range(hyp['optimizer']['n_epochs'])]

    # creating folders for logging losses/acc and network weights
    log_path, net_path = create_folders_logging(net_name)
    print(f'Log_folders: {log_path} -- {net_path}')

    # saving the randomly initialized network
    torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{-1}.pth') # saving random weights

    print('\nTraining begins here!\n')

    for epoch in range(hyp['optimizer']['n_epochs']):

        start = time.time()

        torch.cuda.synchronize()
        
        train_loss_running = 0.0
        train_acc_running = 0.0
        batch = 0

        for images,labels in train_loader:

            imgs = images.to(hyp['optimizer']['device'])
            lbls = labels.to(hyp['optimizer']['device'])

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                loss = criterion(outputs[0], lbls.long())
                if hyp['network']['timesteps'] > 1:
                    for t in range(hyp['network']['timesteps']-1):
                        loss = loss + criterion(outputs[t+1], lbls.long())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_running += loss.item()
            train_loss_running = train_loss_running
            train_acc_running += np.mean(compute_accuracy(outputs,lbls))

            batch += 1
            if show_progress_bar:
                print(f'Training Epoch {epoch}: Batch {batch} of {len(train_loader)}', end="\r")

        print('\nEpoch time: ', "{:.2f}".format(time.time() - start), ' seconds')
            
        train_losses[epoch] = train_loss_running/len(train_loader)
        train_accuracies[epoch] = train_acc_running/len(train_loader)
        
        # getting validation loss and acc
        net.eval()
        val_loss_running, val_acc_running = eval_network(val_loader,net,criterion,hyp)
        net.train()

        val_losses[epoch] = val_loss_running/len(val_loader)
        val_accuracies[epoch] = np.mean(val_acc_running)/len(val_loader)
        
        print(f'Train loss: {train_losses[epoch]:.2f}; acc: {train_accuracies[epoch]:.2f}%')
        print(f'Val loss: {val_losses[epoch]:.2f}; acc: {val_accuracies[epoch]:.2f}%\n')
        
        if (epoch+1) % hyp['misc']['save_logs'] == 0:
            np.savez(log_path+'/loss_'+net_name+'.npz', train_loss=train_losses, val_loss=val_losses, train_accuracies=train_accuracies,
                    val_accuracies=val_accuracies)
        if (epoch+1) % hyp['misc']['save_net'] == 0:
            torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{epoch}.pth')

    print('\n Done training!\n')

    torch.save(net.state_dict(), f'{net_path}/{net_name}.pth') # saving final model

    # getting test loss and acc
    net.eval()
    test_loss_running, test_acc_running = eval_network(test_loader,net,criterion,hyp)
    test_acc = test_acc_running/len(test_loader)
    print(f'Test accuracies over time (%): {test_acc}')