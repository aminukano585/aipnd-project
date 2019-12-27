import torch
from torch import nn
import torch.nn.functional as F

import time
import copy

class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.hidden_sizes = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        
        self.hidden_sizes.extend([
            nn.Linear(h1, h2) for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:])
        ])
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(p=0.35)
        
    def forward(self, x):
        for layer in self.hidden_sizes:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)
    
    
def train_model(model, criterion, optimizer, scheduler, image_datasets, dataloaders, epochs, gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if gpu else "cpu"
    model.to(device)
    
    start = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch, _ in enumerate(range(epochs), start=1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for _, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicted == labels.data)
                
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
                        
            print('{} Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                phase.capitalize(), epoch_loss, epoch_acc*100
            ))
                    
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print('\n')
        
    stop = time.time() - start
    print('Training Completed in {:.0f}m {:.0f}s using {}'.format(
        stop // 60, stop % 60, device
    ))
    print('Best Validation Accuracy achieved is {:.2f}%'.format(best_acc*100))
    
    model.load_state_dict(best_model_wts)
    return model    


def display_test_accuracy(model, dataloaders, gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if gpu else "cpu"
    model.to(device)
    
    model.eval()
    accuracy = 0

    for _, (inputs, labels) in enumerate(dataloaders['test']):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        equality = (labels.data == outputs.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print('Test Accuracy achieved is {:.2f}%'.format(100*accuracy/len(dataloaders['test'])))