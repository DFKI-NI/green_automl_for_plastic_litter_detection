#!/usr/bin/env python
# coding: utf-8

import argparse
from nni.experiment import Experiment

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--model", help="type of model: alexnet, efficient_net, mobile_net, resnet or vgg")
argParser.add_argument("-p", "--path_to_data", help="Path to dataset")
args = argParser.parse_args()

search_space = {
    'lr': {'_type': 'choice', '_value': [0.0001, 0.001, 0.01, 0.1]},
    'weight_decay': {'_type': 'choice', '_value': [0.0001, 0.001, 0.01, 0.1]},
    'dropout': {'_type': 'choice', '_value': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    'number_of_epochs': {'_type': 'choice', '_value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    'batch_size': {'_type': 'choice', '_value': [8, 16, 32, 64]}
}

experiment = Experiment('local')
experiment.config.trial_command = 'python train.py ' + args.model + ' ' + args.path_to_data
experiment.config.trial_code_directory = ''
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.tuner.class_args = {
    'seed': 100
}
experiment.config.max_trial_number = 30
experiment.config.trial_concurrency = 2
experiment.config.experiment_working_directory = 'output_dir'
experiment.config.max_trial_duration = "3h"
experiment.run(8090, debug=True)
experiment.stop()
