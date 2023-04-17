#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle as pckl

import numpy as np
import torch
import torch.nn.utils.prune as prune
from codecarbon import EmissionsTracker
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import alexnet, efficientnet, mobilenetv3, resnet, vgg

from train import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--model", help="type of model: alexnet, efficient_net, mobile_net, resnet or vgg")
argParser.add_argument("-p", "--path_to_data", help="Path to dataset")
argParser.add_argument("-p_t", "--path_to_best_trial", help="Path to best trial")
args = argParser.parse_args()

path_to_best_trial = args.path_to_best_trial
model_name = args.model
path_to_data = args.path_to_data

with open(path_to_best_trial + 'parameter.cfg', 'r+') as file:
    params = dict(eval(file.read()))['parameters']

if model_name == 'alexnet':
    model = alexnet.AlexNet(num_classes=8, dropout=params['dropout']).to(device)
elif model_name == 'efficient_net':
    model = efficientnet.efficientnet_v2_s(num_classes=8).to(device)
elif model_name == 'mobile_net':
    model = mobilenetv3.mobilenet_v3_small(num_classes=8, dropout=params['dropout']).to(device)
elif model_name == 'resnet':
    model = resnet.resnet18(num_classes=8).to(device)
elif model_name == 'vgg':
    model = vgg.vgg11(num_classes=8, dropout=params['dropout']).to(device)
else:
    "Model not defined"

model.load_state_dict(torch.load(path_to_best_trial + '/net.pth'))
model.eval()

label_names = pckl.load(open(path_to_data + '/label_names.pckl', 'rb'))

# read data
data = datasets.ImageFolder(path_to_data, transform=transforms.ToTensor())
train_val_test = pckl.load(open(path_to_data + '/train_val_test.pckl', 'rb'))
ind = np.array(range(0, len(data)))
test = torch.utils.data.Subset(data, ind[(train_val_test == 2)[:, 0]])
test_dataloader = DataLoader(test, batch_size=params['batch_size'], shuffle=True)


def get_metrics(model_in):
    res = dict()
    with EmissionsTracker(tracking_mode='process', log_level='critical', co2_signal_api_token='') as tracker:
        res['test_acc'], res['test_loss'] = evaluate_model(model_in, test_dataloader, cuda=True)
    res['emissions_test'] = tracker.final_emissions
    x = torch.randn(1, 3, 128, 128).to(device)
    res['flops'] = FlopCountAnalysis(model_in, x).total()
    return res


print('Results before pruning', get_metrics(model))

parameters_to_prune = [(module, "weight") for module in
                       filter(lambda m: (type(m) == torch.nn.Linear) | (type(m) == torch.nn.Conv2d), model.modules())]

prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.9, )

print('Results after pruning', get_metrics(model))
pckl.dump(model, open(path_to_best_trial + 'model_after_pruning.pckl', 'wb'))
