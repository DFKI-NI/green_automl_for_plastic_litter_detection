#!/usr/bin/env python
# coding: utf-8

import pickle as pckl
import sys

import nni
import numpy as np
import torch
import torch.optim as optim
from codecarbon import EmissionsTracker
from fvcore.nn import FlopCountAnalysis
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import alexnet, efficientnet, mobilenetv3, resnet, vgg

params = {
    'lr': 0.0001,
    'number_of_epochs': 10,
    'batch_size': 8,
    'weight_decay': 0.01,
    'dropout': 0.1
}

optimized_params = nni.get_next_parameter()  # get parameters from nni optimizer
params.update(optimized_params)
print(params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = sys.argv[1]

# read data
path_dataset = sys.argv[2]
data = datasets.ImageFolder(path_dataset, transform=transforms.ToTensor())
train_val_test = pckl.load(open(path_dataset + '/train_val_test.pckl', 'rb'))
ind = np.array(range(0, len(data)))
train = torch.utils.data.Subset(data, ind[(train_val_test == 0)[:, 0]])
val = torch.utils.data.Subset(data, ind[(train_val_test == 1)[:, 0]])
test = torch.utils.data.Subset(data, ind[(train_val_test == 2)[:, 0]])

train_dataloader = DataLoader(train, batch_size=params['batch_size'], shuffle=True)
val_dataloader = DataLoader(val, batch_size=params['batch_size'], shuffle=True)
test_dataloader = DataLoader(test, batch_size=params['batch_size'], shuffle=True)

# define the model

if model_name == 'alexnet':
    net = alexnet.AlexNet(num_classes=8, dropout=params['dropout']).to(device)
elif model_name == 'efficient_net':
    net = efficientnet.efficientnet_v2_s(num_classes=8).to(device)
elif model_name == 'mobile_net':
    net = mobilenetv3.mobilenet_v3_small(num_classes=8, dropout=params['dropout']).to(device)
elif model_name == 'resnet':
    net = resnet.resnet18(num_classes=8).to(device)
elif model_name == 'vgg':
    net = vgg.vgg11(num_classes=8, dropout=params['dropout']).to(device)
else:
    "Model not defined"
optimizer = optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])


def evaluate_model(model, loader, cuda=False):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = total = 0
        loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            correct += (predict == targets).sum().cpu().item()
            loss += criterion(logits, targets)
            total += targets.size(0)
    print('Accuracy:', correct / total)
    acc = correct / total
    loss /= len(loader)
    return acc, loss.item()


criterion = nn.CrossEntropyLoss()
# add API token if available
with EmissionsTracker(tracking_mode='process', log_level='critical', co2_signal_api_token='') as tracker:
    for epoch in range(params['number_of_epochs']):
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

emissions_train = tracker.final_emissions

print('Finished Training')
current_trial = 'output_dir/' + nni.get_experiment_id() + '/' + 'trials/' + nni.get_trial_id()
torch.save(net.state_dict(), current_trial + '/net.pth')

with EmissionsTracker(tracking_mode='process', log_level='critical', co2_signal_api_token='') as tracker:
    val_acc, val_loss = evaluate_model(net, val_dataloader, cuda=True)
emissions_val = tracker.final_emissions

x = torch.randn(1, 3, 128, 128).to(device)
flops = FlopCountAnalysis(net, x).total()

nni.report_final_result({'default': val_acc, 'val_acc': val_acc, 'val_loss': val_loss, 'flops': flops,
                         'emissions_val': 'emissions_val', 'emissions_train': emissions_train})
