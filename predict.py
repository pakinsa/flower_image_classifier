import 



load_checkpoint(filepath)

optimizer, model = load_checkpoint('checkpoint.pth')
print(model)
#print(optimizer)
print("new optimizer loaded successfully on")




import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import seaborn as sb

import argparse
import os


def main():

    args = args_parser()
    print(args) 



def basic_model(arch):
    # Load pretrained_network
    if arch == None or arch == 'vgg16':
        load_model = models.vgg16(pretrained = True)
        #load_model.name = 'vgg16'
        print('Current model: vgg16 loaded')
    else:
        load_model = models.resnet18 (pretrained = True)
        print('Current model: resnet18 loaded')
        #model.name = arch
    
    return load_model

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(file, arch):
    
    # Load checkpoint according to filename/filepath
    checkpoint = torch.load(file)
    
    # Load model corresponding to arch - type
    model = basic_model(arch)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.mapping = checkpoint['mapping']
    model.number_of_epochs = checkpoint['number_of_epochs']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)    
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    return optimizer, model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    np_image = transform(pil_image)
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu==True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    
    # Image needs to be preprocessed: use the process_image function defined above
    image = process_image(image_path)
   
    # Image from nd.array to tensor
    #image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    model_input = image.unsqueeze(0)
    model_input = model_input.to(device)
 
    #Let's pass the image in our model and get the probabilities 
    output = model.forward(model_input)
    probabilities = torch.exp(output)
    
    #Top 5 probabilities and corresponding indices 
    topk_prob, topk_indices = probabilities.topk(topk)
    
    #Converting topk_prob and topk_classes to lists, so that we can loop through them without errors
    top_probabilities = topk_prob[0].tolist() 
    top_indices = topk_indices[0].tolist()
    
    #Inverted dictionary: mapping from index to class
    inverted_dic = {index:category for category, index in model.mapping.items()}
    
    top_numbers= [inverted_dic[element]for element in top_indices]
    top_labels= [cat_to_name[str(k)] for k in top_numbers]
    
    return top_probabilities, top_numbers, top_labels

def main():
    
    args = args_paser()
    
    # Definition of image data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Load Checkpoint
    optimizer, load_model = load_checkpoint('checkpoint.pth', args.arch)
    
    # Test if image processing works
    image_path = test_dir + '/43/' + 'image_02431.jpg'
    #processed_image = (test_dir + '/47/' + 'image_04966.jpg')
    processed_image = process_image(image_path)
    print(processed_image.shape)
    
    top_probabilities, top_numbers, top_labels = predict(image_path, load_model, args.gpu)
    print(top_probabilities)
    print(top_numbers)
    print(top_labels)
        
if __name__ == '__main__': main()



https://knowledge.udacity.com/questions/708597
https://github.com/tcyw2910/Train.py/blob/main/predict%20%281%29.py