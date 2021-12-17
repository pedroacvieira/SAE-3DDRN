#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:33 2021

@author: Pedro Vieira
@description: Implements the test function for the SAE-3DDRN network adding noise to the input image
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from utils.config import SAE3DConfig
from utils.dataset import DRNDataset
from utils.tools import *
from net.sae import SAE
from net.drn import DRN
from utils.noise import add_noise
from test import test_model

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
CONFIG_FILE = ''  # Empty string to load default 'config.yaml'
NOISE_TYPES = ['salt_and_pepper', 'additive_gaussian', 'multiplicative_gaussian']


# Test SAE-3DDRN runs
def test():
    # Load config data from training
    config_file = 'config.yaml' if not CONFIG_FILE else CONFIG_FILE
    cfg = SAE3DConfig(config_file, test=True)

    # Start tensorboard
    writer = None
    if cfg.use_tensorboard:
        writer = SummaryWriter(cfg.tensorboard_folder)

    # Set string modifier if testing best models
    test_best = 'best_' if cfg.test_best_models else ''
    if cfg.test_best_models:
        print('Testing best models from each run!')

    # Load processed dataset
    data = torch.load(cfg.exec_folder + 'proc_image.pth')

    for run in range(cfg.num_runs):
        print(f'TESTING STACKED AUTOENCODER FROM RUN {run + 1}/{cfg.num_runs}')
        _, test_gt, _ = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)
        test_gt = HSIData.remove_negative_gt(test_gt)
        num_classes = len(np.unique(test_gt)) - 1

        # Load autoencoder model
        sae_file = cfg.exec_folder + f'runs/sae_model_run_' + str(run) + '.pth'
        sae = SAE(data.shape[2], cfg.sae_hidden_layers)
        sae.load_state_dict(torch.load(sae_file))
        sae.eval()

        for noise_type in NOISE_TYPES:
            noisy_data = add_noise(data, noise_type)
            sae_noisy_data = sae(noisy_data)

            print(f'TESTING SAE-3DDRN MODEL FROM RUN {run + 1}/{cfg.num_runs}')
            # Load test ground truth and initialize test loader
            test_dataset = DRNDataset(sae_noisy_data, test_gt, cfg.sample_size, data_augmentation=False)
            test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

            # Load model
            model_file = cfg.exec_folder + f'runs/drn_{test_best}model_run_' + str(run) + '.pth'
            model = nn.DataParallel(DRN(num_classes))
            model.load_state_dict(torch.load(model_file))
            model.eval()

            # Set model to device
            model = model.to(device)

            # Test model from the current run
            report = test_model(model, test_loader, writer)
            filename = cfg.results_folder + f'test_{noise_type}.txt'
            save_results(filename, report, None, run)

    if cfg.use_tensorboard:
        writer.close()


# Main for running test independently
def main():
    test()


if __name__ == '__main__':
    main()
