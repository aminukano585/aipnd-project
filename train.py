import argparse

import torch
from torch import nn, optim

import utilities
import classifiers


def main():
    parser = argparse.ArgumentParser(
        description='Image Classifier Command Line App\nTrain a Model'
    )
    
    parser.add_argument('data_dir', type=str, default='flowers',
                        help='Specify the image data directory e.g python train.py flowers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Specify the checkpoints directory e.g --save_dir checkpoints')
    parser.add_argument('--arch', type=str, default='densenet161',
                        help='Specify the model architecture e.g --arch densenet\nother options resnet, vgg, alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Specify the network learning rate e.g --learning_rate 0.001')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[512],
                        help='Specify the number of hidden layers e.g --hidden_layers 512 256 128')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Specify number of epochs e.g --epochs 10')
    parser.add_argument('--gpu', action='store_true',
                        help='Specify whether to use GPU e.g --gpu')
    
    args = parser.parse_args()
    train(args)
    
    
def train(args):
    data_dir = args.data_dir
    arch = args.arch
    learning_rate = args.learning_rate
    epochs = args.epochs
    gpu = args.gpu
    save_dir = args.save_dir
    
    image_datasets, dataloaders = utilities.preprocess_images(data_dir)
    model = utilities.get_torch_model(arch)
    
    for param in model.parameters():
        param.requires_grad = False
        
    input_size = utilities.get_input_size(model, arch)
    hidden_sizes = args.hidden_units
    output_size = 102
    
    model.classifier = classifiers.Network(input_size, hidden_sizes, output_size)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4)
    
    model = classifiers.train_model(model, criterion, optimizer, scheduler, image_datasets, dataloaders, epochs, gpu)
    
    classifiers.display_test_accuracy(model, dataloaders, gpu)
    utilities.save_checkpoint(model, optimizer, scheduler, image_datasets, arch, save_dir)
    
    
if __name__ == '__main__':
    main()