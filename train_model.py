from collections import OrderedDict
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import  models, datasets
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from transformer import transformer
import torch
from torch.utils.data import DataLoader



def train_model(dataloaders_tr, dataloaders_va, hidden_units1, hidden_units2, epochs, arch, lr, dev):
    
      
    
    if arch=='densenet121' or arch==None:
        
        
        model = models.densenet121(pretrained=True)


        # frezing ImageNet models' features parameters
        for p in model.parameters():
            p.requires_grad=False


        # Defining an new but untrained classifier model
        classifier = nn.Sequential(OrderedDict([('fcl1', nn.Linear(1024, hidden_units1)),
                                            ('relu1', nn.ReLU()),
                                            ('drop1', nn.Dropout(0.2)),
                                            ('fcl2', nn.Linear(hidden_units1, hidden_units2)),
                                            ('relu2', nn.ReLU()),
                                            ('fcl3', nn.Linear(hidden_units2, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
                                  
        model.classifier = classifier
    
        criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        # default for this model is 0.003, can be further lower to 0.0003
        
        
        
        print("...setting up model for training...")
        for device in ['cpu', 'cuda']:
    
            if torch.cuda.is_available() and dev=='gpu':
                # if(dev=='gpu'):
                map_location=lambda storage, loc: storage.cuda(), 'cuda'
                print(dev, 'is available and', map_location, 'in use')
                # Set default tensor type, then move model and data to GPU
                # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
            else:
                map_location='cpu'
                print(dev, 'is available but', map_location, 'in use')
                       
            
            model.to(device)

            train_losses = []
            valid_losses = []
            epochs = epochs    # No of epochs here is 3
            steps = 0
            training_loss = 0
            print_every = 5
           
            print("hyperparameters for training are:", "arch:", arch,  \
                  "epochs:", epochs,  "lr:", lr,  "hidden_units1:", hidden_units1,  \
                  "hidden_units2:", hidden_units2,  "criterion:", criterion,  "optimizer:", optimizer)
            print("...now training model on", map_location)
           
           
            for epoch in range(epochs):
        
                for inputs, labels in dataloaders_tr:
                    steps += 1
                    # Move data and label tensors to the default device
                    inputs, labels = inputs.to(device), labels.to(device)
        
                    yhat1 = model.forward(inputs)
                    loss = criterion(yhat1, labels)
        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    training_loss += loss.item()
        
                    if steps % print_every == 0:
                        validating_loss = 0
                        accuracy = 0
                        model.eval()
                        with torch.no_grad():
                            for inputs, labels in dataloaders_va:
                                inputs, labels = inputs.to(device), labels.to(device)
                                yhat2 = model.forward(inputs)
                                batch_loss = criterion(yhat2, labels)
                    
                                validating_loss += batch_loss.item()
                    
                                # Calculate accuracy
                                ps = torch.exp(yhat2)
                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                                train_losses.append(training_loss/len(dataloaders_tr))
                                valid_losses.append(validating_loss/len(dataloaders_va))
                    
                        print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {training_loss/print_every:.3f}.. "
                              f"Validation loss: {validating_loss/len(dataloaders_va):.3f}.. "
                              f"Validation accuracy: {accuracy/len(dataloaders_va):.3f}")
                        training_loss = 0
                        model.train()
                

                
    elif(arch=='alexnet'):
        
        model = models.alexnet(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False


        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, hidden_units1)),
                          ('relu1', nn.ReLU()),
                          ('drop1',nn.Dropout(0.3)),
                          ('fc2', nn.Linear(hidden_units1, hidden_units2)),
                          ('relu2',nn.ReLU()),
                          ('drop2',nn.Dropout(0.3)),
                          ('fc3', nn.Linear(hidden_units2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.classifier = classifier

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        # default lr for this model is 0.0001. 
        # model.to(device);
        
        
        print("...setting up model for training...")
        for device in ['cpu', 'cuda']:
        
            if torch.cuda.is_available() and dev=='gpu':
                # if(dev=='gpu'):
                map_location=lambda storage, loc: storage.cuda(), 'cuda'
                print(dev,'is available and', map_location, 'in use')
                #Set default tensor type, then move model and data to GPU
                #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                map_location='cpu'
                print(dev,'is available but', map_location, 'in use')
                
            model.to(device)
    
            train_losses = []
            valid_losses = []
            epochs = epochs    # default No of epochs here is 10
            steps = 0
            training_loss = 0
            
            print("hyperparameters for training are:", "arch:", arch,  \
                  "epochs:", epochs,  "lr:", lr,  "hidden_units1:", hidden_units1,  \
                  "hidden_units2:", hidden_units2,  "criterion:", criterion,  "optimizer:", optimizer)
            print("...now training model on", map_location)
            
            for epoch in range(epochs):
                 
               for inputs, labels in dataloaders_tr:
                   steps+=1
                   inputs, labels = inputs.to(device), labels.to(device)
                   model.to(device)
                   
                   optimizer.zero_grad()
        
                   yhat1 = model.forward(inputs)
                   loss = criterion(yhat1, labels)
                   loss.backward()
                   optimizer.step()
        
                    
                   training_loss +=loss.item()
            
                   if steps % 5 == 0:
                       validation_loss = 0
                       accuracy = 0
                       model.eval()
        
                       with torch.no_grad():
            
                           for inputs, labels in dataloaders_va:
                               inputs, labels = inputs.to(device), labels.to(device)
                            
                               yhat2= model(inputs)
                               batch_loss = criterion(yhat2, labels)
                
                               validation_loss+=batch_loss.item()
                    
                               ps = torch.exp(yhat2)
                               top_prob, top_class = ps.topk(1, dim=1)
                               equals = top_class == labels.view(*top_class.shape)
                               accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                           train_losses.append(training_loss/len(dataloaders_tr))
                           valid_losses.append(validation_loss/len(dataloaders_va))
        
                       

                       print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                          "Training Loss: {:.3f}.. ".format(training_loss/len(dataloaders_tr)),
                          "Valid Loss: {:.3f}.. ".format(validation_loss/len(dataloaders_va)),
                          "Valid Accuracy: {:.3f}".format(accuracy/len(dataloaders_va)))
                
                                     
               