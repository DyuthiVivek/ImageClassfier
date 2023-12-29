import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

def transform_load_data(data_dir = 'flowers'):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    return trainloader, validloader, testloader, train_dataset, valid_dataset, test_dataset



def model_setup(name = 'densenet121', hidden_units = 256, learn_rate = 0.003, dropout = 0.5, use_gpu = True):
    if name == 'vgg16':
        model = models.vgg16(pretrained = True)
        first_layer = 25088
    elif name == 'densenet121':
        model = models.densenet121(pretrained = True)
        first_layer = 1024
    elif name == 'alexnet':
        model = models.alexnet(pretrained = True)
        first_layer = 9216

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(first_layer, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(256, 90),
                                     nn.ReLU(),
                                     nn.Linear(90, 80),
                                     nn.ReLU(),
                                     nn.Linear(80, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    
    if use_gpu and torch.cuda.is_available():
        model.cuda()

    return model, criterion, optimizer

def train_model(trainloader, validloader, model, criterion, optimizer, epochs = 2, use_gpu = True):
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            
            if torch.cuda.is_available() and use_gpu:
                inputs,labels = inputs.to('cuda'), labels.to('cuda')

            steps += 1
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        if torch.cuda.is_available() and use_gpu:
                            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {test_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

def save_checkpoint(model, train_dataset, arch = 'densenet121', lr = 0.003, epochs = 2, dropout = 0.5, hidden_units = 256, path = 'checkpoint.pth'):
    model.class_to_idx = train_dataset.class_to_idx
    model.cpu
    torch.save({'structure' : arch,
                'hidden_layer1': hidden_units,
                'lr': lr,
                'epochs': epochs,
                'dropout' : dropout,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
            
def load_checkpoint(path = 'checkpoint.pth'):
    checkpoint = torch.load(str(path))
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    lr = checkpoint['lr']
    drop = checkpoint['dropout']
    model, criterion, optimizer = model_setup(name = structure, hidden_units = hidden_layer1, learn_rate = lr, dropout = drop)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model, criterion, optimizer

def process_image(image):
    img_pil = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(img_pil)


def predict(image_path, model, topk = 5, use_gpu = True):
    if torch.cuda.is_available() and use_gpu:
        model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if use_gpu:
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)
        
    probability = F.softmax(output.data, dim = 1)
    return probability.topk(topk)
