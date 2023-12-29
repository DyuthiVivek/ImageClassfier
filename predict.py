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
import json

# Note: all functions in utils.py

arg = argparse.ArgumentParser(description = 'predict-file')
arg.add_argument('data_dir', nargs = '*', action = "store", default = "./flowers/")
arg.add_argument('image_path', default = 'flowers/test/10/image_07090.jpg', nargs = '*', action = "store", type = str)
arg.add_argument('--model_path', default = 'checkpoint.pth', dest = 'model_path', nargs = '*', action = 'store', type = str)
arg.add_argument('--gpu', default = "gpu", action = "store", dest = "gpu")
arg.add_argument('--top_k', default = 5, dest = "top_k", action = "store", type = int)
arg.add_argument('--category_names', dest = "category_names", action = "store", default = 'cat_to_name.json')

parsed = arg.parse_args()
image_path = parsed.image_path
use_gpu = (parsed.gpu == "gpu")
top_k = parsed.top_k
model_path = parsed.model_path
data_dir = parsed.data_dir
category_names = parsed.category_names

trainloader, validloader, testloader, train_dataset, valid_dataset, test_dataset = utils.transform_load_data(data_dir)

model, criterion, optimizer = utils.load_checkpoint(model_path)

with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file, strict=False)


probs, classes = utils.predict(image_path, model, top_k, use_gpu)

class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
classes = [class_to_idx_inverted[i] for i in classes.cpu().numpy()[0]]
probs = probs.cpu().numpy()[0]

flower_classes = [cat_to_name[j] for j in classes]

print("Classes and probability:")
for i in range(top_k):
    print(flower_classes[i], "with probability", probs[i])