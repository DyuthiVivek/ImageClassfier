import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import utils

# Note: all functions in utils.py

arg = argparse.ArgumentParser()
arg.add_argument('data_dir', nargs = '*', action = "store", default = "./flowers/")
arg.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu")
arg.add_argument('--save_dir', dest = 'save_dir', action = "store", default = "./checkpoint.pth")
arg.add_argument('--epochs', dest = "epochs", action = "store", type = int, default = 1)
arg.add_argument('--arch', dest = "arch", action = "store", default = 'densenet121', type = str)
arg.add_argument('--learning_rate', dest = "learning_rate", action = 'store', default = 0.003)
arg.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
arg.add_argument('--hidden_units', type = int, dest = "hidden_units", action = "store", default = 256)

parsed = arg.parse_args()
data_dir = parsed.data_dir
path = parsed.save_dir
arch = parsed.arch
dropout = parsed.dropout
lr = parsed.learning_rate
hidden_layer1 = parsed.hidden_units
use_gpu = (parsed.gpu == 'gpu')
epochs = parsed.epochs

trainloader, validloader, testloader, train_dataset, valid_dataset, test_dataset = utils.transform_load_data(data_dir)
model, criterion, optimizer = utils.model_setup(name = arch, hidden_units = hidden_layer1, learn_rate = lr, dropout = dropout, use_gpu = use_gpu)
utils.train_model(trainloader, validloader, model, criterion, optimizer, epochs = epochs, use_gpu = use_gpu)
utils.save_checkpoint(model, train_dataset, arch = arch, lr = lr, epochs = epochs, dropout = dropout, hidden_units = hidden_layer1, path = 'checkpoint.pth')

