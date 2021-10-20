import torch
from torchvision import datasets, transforms, models

def save_checkpoint(model, image_datasets_tr, save_dir):
    
    '''This function saves the trained model as checkpoint'''
    
    # model.class_to_idx = trainset.class_to_idx
    
    checkpoint = {'arch': arch,
              'epoch': epochs,
              'learning_rate': lr,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets_tr.class_to_idx,
              'optimizer_dict': optimizer.state_dict()}
   
    print("Now lets save Checkpoint !!!")
    
    return torch.save(checkpoint, save_dir)


