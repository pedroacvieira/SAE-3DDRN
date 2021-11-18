#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 09 17:44 2021

@author: Pedro Vieira
@description: Settings for the SAE-3DDRN training and testing
"""

import shutil
import os
import yaml


class SAE3DConfig:
    def __init__(self, filename='config.yaml', test=False):
        # Load config
        with open(filename, "r") as file:
            cfg = yaml.safe_load(file)

            # Dataset settings
            self.dataset = cfg['dataset']
            self.experiment = cfg['experiment']
            self.data_folder = cfg['data_folder']
            self.exec_folder = cfg['exec_folder'] + self.experiment + '/'
            self.split_folder = self.exec_folder + cfg['split_folder'] + self.dataset + '/'
            self.val_split = cfg['val_split']
            self.train_split = cfg['train_split']
            self.generate_samples = cfg['generate_samples']
            self.max_samples = cfg['max_samples']

            # Hyper parameters
            self.train_batch_size = cfg['train_batch_size']
            self.test_batch_size = cfg['test_batch_size']
            self.sample_size = cfg['sample_size']
            self.num_runs = cfg['num_runs']
            self.num_epochs = cfg['num_epochs']
            self.learning_rate = cfg['learning_rate']
            self.betas = tuple(cfg['betas'])
            self.weight_decay = float(cfg['weight_decay'])
            self.gamma = cfg['gamma']
            self.scheduler_step = cfg['scheduler_step']
            self.drop_out = cfg['drop_out']

            # Stacked autoencoder parameters
            self.sae_hidden_layers = cfg['sae_hidden_layers']
            self.sae_train_batch_size = cfg['sae_train_batch_size']
            self.sae_test_batch_size = cfg['sae_test_batch_size']
            self.sae_num_epochs = cfg['sae_num_epochs']
            self.sae_learning_rate = cfg['sae_learning_rate']
            self.sae_betas = tuple(cfg['sae_betas'])
            self.sae_weight_decay = float(cfg['sae_weight_decay'])
            self.sae_gamma = cfg['sae_gamma']
            self.sae_scheduler_step = cfg['sae_scheduler_step']

            # Other options
            self.test_best_models = cfg['test_best_models']
            self.use_checkpoint = cfg['use_checkpoint']
            self.results_folder = self.exec_folder + cfg['results_folder']
            self.checkpoint_folder = self. exec_folder + cfg['checkpoint_folder'] + self.dataset + '/'
            self.checkpoint_file = cfg['checkpoint_file']
            self.print_frequency = cfg['print_frequency']
            self.sae_print_frequency = cfg['sae_print_frequency']
            self.use_tensorboard = cfg['use_tensorboard']
            if self.use_tensorboard:
                self.tensorboard_folder = self.exec_folder + 'tensorboard/'

        # Copy config to execution folder
        if not (test or self.use_checkpoint or not self.generate_samples):
            assert not os.path.isdir(self.exec_folder), 'Current experiment name already exists. '\
                                                        'Please provide a new experiment name.'
            os.makedirs(self.exec_folder)
            os.makedirs(self.split_folder)
            os.makedirs(self.results_folder)
            os.makedirs(self.checkpoint_folder)
            os.makedirs(self.exec_folder + 'runs/')
            if self.use_tensorboard:
                os.makedirs(self.tensorboard_folder)
            shutil.copyfile(filename, self.exec_folder + filename)
