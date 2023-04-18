#!/usr/bin/env python
# coding: utf-8


import pickle as pckl

import numpy as np
import torch
from codecarbon import EmissionsTracker
from fvcore.nn import FlopCountAnalysis
from nni.retiarii import fixed_arch
from nni.retiarii.evaluator.pytorch import Classification
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.hub.pytorch import ENAS
from nni.retiarii.strategy import ENAS as ENAS_startegy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from standard_architectures.train import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--path_to_data", help="Path to dataset")
args = argParser.parse_args()

# read data
path_to_data = args.path_to_data
data = datasets.ImageFolder(path_to_data, transform=transforms.ToTensor())
train_val_test = pckl.load(open(path_to_data + '/train_val_test.pckl', 'rb'))
ind = np.array(range(0, len(data)))
train = torch.utils.data.Subset(data, ind[(train_val_test == 0)[:, 0]])
val = torch.utils.data.Subset(data, ind[(train_val_test == 1)[:, 0]])
test = torch.utils.data.Subset(data, ind[(train_val_test == 2)[:, 0]])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=True)

# train
model_space = ENAS(
    width=32,  # the initial filters (channel number) for the model
    num_cells=(3, 3),  # the number of stacked cells in total
    dataset='imagenet'  # to give a hint about input resolution, here is 32x32
)

evaluator = Classification(
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_epochs=100,
    gpus=1,
    fast_dev_run=False
)

strategy = ENAS_startegy(reward_metric_name='train_loss')

config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator, [], strategy)

# add API token if available
with EmissionsTracker(tracking_mode='process', log_level='critical', co2_signal_api_token='') as tracker:
    # find best architecture
    experiment.run(config)
    experiment.stop()

    # Train best model
    model_space.eval()
    exported_arch = experiment.export_top_models()[0]

    with fixed_arch(exported_arch):
        final_model = ENAS(width=32, num_cells=(3, 3), dataset='imagenet')

    evaluator = Classification(
        learning_rate=1e-3,
        weight_decay=1e-4,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_epochs=100,
        gpus=1,
        fast_dev_run=False
    )

    evaluator.fit(final_model)

emissions_train = tracker.final_emissions

print('Finished Training')

# add API token if available
with EmissionsTracker(tracking_mode='process', log_level='critical', co2_signal_api_token='') as tracker:
    test_acc, test_loss = evaluate_model(final_model, test_dataloader)
emissions_test = tracker.final_emissions

x = torch.randn(1, 3, 128, 128).to(device)
flops = FlopCountAnalysis(final_model, x).total()

print('Test Loss: ', test_loss)
print('Test accuracy: ', test_acc)
print('Test emissions: ', emissions_test)
print('Train emissions: ', emissions_train)
print('flops: ', flops)
