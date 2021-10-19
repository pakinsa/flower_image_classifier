import torch

def save_checkpoint(model, image_datasets_tr, save_dir):
    
    '''This function saves the trained model as checkpoint'''
    
       
    new_point = {'arch': arch,
                  'epoch': epochs,
                  'learning_rate': lr,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': image_datasets_tr.class_to_idx,
                  'optimizer_dict': optimizer.state_dict()}
    
    return torch.save(new_point, save_dir)