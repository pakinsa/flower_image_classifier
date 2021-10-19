from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim

import torch


def transformer(train_dir, test_dir, valid_dir):
    ''' This function loads the dataset, transforms to acceptable
        input format for the neural network. And returns the 
        respective dataloaders
    ''' 
   
    
    data_tr_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), 
                                                          (0.229, 0.224, 0.225))])

    data_va_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(244),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_te_transforms = transforms.Compose ([transforms.Resize(255),
                                         transforms.CenterCrop(244),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets_tr = datasets.ImageFolder(train_dir, transform=data_tr_transforms)
    image_datasets_va = datasets.ImageFolder(valid_dir, transform=data_va_transforms)
    image_datasets_te = datasets.ImageFolder(test_dir, transform=data_te_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders_tr = torch.utils.data.DataLoader(image_datasets_tr, batch_size=256, shuffle=True)
    dataloaders_va = torch.utils.data.DataLoader(image_datasets_va, batch_size=256, shuffle=False)
    dataloaders_te = torch.utils.data.DataLoader(image_datasets_te, batch_size=256, shuffle=False)
     
    return dataloaders_tr, dataloaders_va, dataloaders_te

