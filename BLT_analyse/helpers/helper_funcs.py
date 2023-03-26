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

##############################
## Loading the dataset loaders
##############################

def get_Dataset_loaders(hyp):

    # create Datasets for the splits
    if hyp['dataset']['name'] == 'miniecoset':

        dataset_path = hyp['dataset']['dataset_path'] + 'miniecoset_' + str(hyp['dataset']['image_size']) + 'px.h5'

        with h5py.File(dataset_path, "r") as f:
            hyp['dataset']['train_img_mean_channels'] = f['train_img_mean_channels'][()]
            hyp['dataset']['train_img_std_channels'] = f['train_img_std_channels'][()]

        # import the transforms (augmentations)
        transform = get_transform(hyp['dataset']['augment'],hyp)
        transform_val_test = get_transform(['normalize'],hyp)

        train_data = MiniEcoset('train', dataset_path=dataset_path, transform=transform)
        val_data = MiniEcoset('val', dataset_path=dataset_path, transform=transform_val_test)
        test_data = MiniEcoset('test', dataset_path=dataset_path, transform=transform_val_test)
        testplus_data = MiniEcoset('testplus', dataset_path=dataset_path, transform=transform_val_test)

    # create Dataloaders for the splits
    if hyp['optimizer']['device']:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=hyp['optimizer']['batch_size'], shuffle=True,
                                                   num_workers=hyp['optimizer']['dataloader']['num_workers_train'],
                                                   prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_train'])
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=hyp['optimizer']['batch_size'],
                                                 num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                 prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyp['optimizer']['batch_size'],
                                                 num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                 prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
        testplus_loader = torch.utils.data.DataLoader(testplus_data, batch_size=hyp['optimizer']['batch_size'],
                                                      num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                      prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
        
    return train_loader, val_loader, test_loader, testplus_loader
    

def get_transform(aug_str,hyp=None):
    # Returns a transform compose function given the transforms listed in "aug_str"

    transform_list = []
    if 'trivialaug' in aug_str:
        transform_list.append(transforms.TrivialAugmentWide())
    if 'randaug' in aug_str:
        transform_list.append(transforms.RandAugment())
    transform_list.append(transforms.ConvertImageDtype(torch.float))
    if 'normalize' in aug_str:
        transform_list.append(transforms.Normalize(mean = hyp['dataset']['train_img_mean_channels']/255., std = hyp['dataset']['train_img_std_channels']/255.))

    transform = transforms.Compose(transform_list)
    
    return transform


class MiniEcoset(torch.utils.data.Dataset):
    #Import MiniEcoset as a Dataset splitwise

    def __init__(self, split, dataset_path, transform=None):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
        """
        self.root_dir = dataset_path
        self.transform = transform

        with h5py.File(dataset_path, "r") as f:
            self.images = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
            self.labels = torch.from_numpy(f[split]['labels'][()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # accepts ids and returns the images and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.images[idx]
        labels = self.labels[idx]

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels
    
##############################
## Logging functions
##############################
    
def get_folders_logging(net_name):

    print('Accessing log folders...')

    log_folder = '../BLT_train/logs/perf_logs'
    net_folder = '../BLT_train/logs/net_params'

    isExist = os.path.exists(log_folder)
    if not isExist:
        print('Log folder does not exist!')
    isExist = os.path.exists(net_folder)
    if not isExist:
        print('Net folder does not exist!')

    log_folder_name = log_folder+f'/{net_name}'
    net_folder_name = net_folder+f'/{net_name}'

    isExist = os.path.exists(log_folder_name)
    if not isExist:
        print('Specific log folder does not exist!')
    isExist = os.path.exists(net_folder_name)
    if not isExist:
        print('Specific net folder does not exist!')

    save_actvs_folder = 'saved_actvs'
    isExist = os.path.exists(save_actvs_folder)
    if not isExist:
        os.makedirs(save_actvs_folder)
        print('Save actvs folder created!')
    
    save_actvs_folder_name = save_actvs_folder+f'/{net_name}'
    isExist = os.path.exists(save_actvs_folder_name)
    if not isExist:
        os.makedirs(save_actvs_folder_name)
        print('Specific save actvs folder created!')

    return log_folder_name, net_folder_name, save_actvs_folder_name