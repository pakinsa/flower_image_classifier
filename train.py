# Imports here

from time import time
from collections import OrderedDict
from torch.optim import SGD
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from torch import __version__
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim


import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import torch

from select_input_types import args_parser
from transformer import transformer
from train_model import train_model
from save_checkpoint import save_checkpoint

'''This Program trains a machine learning model, with atleast 2 
   torchvision models to predict flowers images'''


def main():

    start_time = time
   

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
        
    args = args_parser()
    print(args) 
    
    
    dataloaders_tr, dataloaders_va, dataloaders_te = transformer(train_dir, test_dir, valid_dir)

    
    print("...model loading....")
    trained_model = train_model(dataloaders_tr, dataloaders_va, \
                                args.hidden_units1, args.hidden_units2, args.epochs, args.arch, args.lr, args.gpu)
    
    
    save_checkpoint(trained_model, image_datasets_tr, args.save_dir)
    print('Checkpoint Saved!')
    
    
    end_time = time

    tot_time = end_time - start_time  #calculate difference between end time and start time
    
    print('Total time taken is:', tot_time)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()

        
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
    print("redefine your hyperparameters by changing the values of this command:"\
          "python train.py  --data_dir flowers  -- epochs 10  --gpu True  --arch vgg16"\
          "--hidden_units1 5024  --hidden_units2 1024  --learning_rate 0.0001")
    
