import torch
from torch import nn, optim
from torchvision import transforms, datasets, models

import numpy as np

import classifiers

import os
import warnings
from PIL import Image
import json
from tabulate import tabulate


def preprocess_images(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in data_transforms.keys()}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                  shuffle=True, drop_last=False)
                      for x in image_datasets.keys()}
    
    return image_datasets, dataloaders


def get_torch_model(arch):
    return getattr(models, arch)(pretrained=True)


def get_input_size(model, arch):
    if ('densenet' in arch): return model.classifier.in_features
    elif ('vgg' in arch): return model.classifier[0].in_features
    elif ('squeezenet' in arch): return model.classifier[1].in_channels
    elif (('resnet' in arch) or ('inception'in arch) ): return model.fc.in_features
    elif ('alexnet' in arch): return model.classifier[1].in_features
    
    
def save_checkpoint(model, optimizer, scheduler, image_datasets, arch, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    filename = 'checkpoint_v1.pth'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        'input_size': model.classifier.hidden_sizes[0].in_features,
        'hidden_sizes': [layer.out_features for layer in model.classifier.hidden_sizes],
        'output_size': model.classifier.output.out_features,
        'arch': arch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save(checkpoint, os.path.join(save_dir, filename))
        
        
def load_checkpoint(checkpoint, gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if gpu else "cpu"
    
    checkpoint = torch.load(checkpoint)
    model = get_torch_model(checkpoint['arch'])
    model.classifier = classifiers.Network(
        checkpoint['input_size'], checkpoint['hidden_sizes'], checkpoint['output_size']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model = model.float().to(device)
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']


def process_image(image):
    val = (256 - 224) * .5 # to get the center crop param
    pil_image = Image.open(image).resize((256, 256)).crop((val, val, 256-val, 256-val))
    np_image = np.array(pil_image) / 255
    norm_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    return norm_image.transpose(2,0,1)


def get_prediction(image, model, top_k, gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if gpu else "cpu"

    if device != "cpu":
        model.cuda()
    model.eval()

    image = process_image(image)
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()
    if device != "cpu":
        image = image.cuda()

    inputs = image.to(device)
    logits = model.forward(inputs)

    probs = nn.functional.softmax(logits, dim=1)
    top_k = probs.cpu().topk(top_k)

    return [i.data.numpy().squeeze() for i in top_k]


def mapping(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name


def get_classes(classes, class_to_idx, category_names):
    idx_to_class = {idx: pic for pic, idx in class_to_idx.items()}
    
    cat_to_name = mapping(category_names)

    return [cat_to_name[idx_to_class[i]] for i in classes]
    
    
def display_result(probs, classes, img_path, category_names, checkpoint, top_k):
    cat_to_name = mapping(category_names)
    input_image = cat_to_name[img_path.split('/')[-2]]
    table_data = [list(i) for i in list(zip(classes, probs))]
    
    print('\n\nImage predicted using {} architecture'.format(torch.load(checkpoint)['arch'].upper()))
    print('-' * 50)
    print('\nInput Image: {}\n'.format(input_image.upper()))
    
    print(tabulate(table_data, headers=['TopK Image Classes', 'Probabilities'], tablefmt='orgtbl'))
    