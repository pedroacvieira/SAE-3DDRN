#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:33 2021

@author: Pedro Vieira
@description: Implements the test function for the SAE-3DDRN network adding noise to the input image
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from net.drn import DRN
from net.sae import SAE
from test import test_model
from utils.config import SAE3DConfig
from utils.dataset import DRNDataset
from utils.noise import add_noise
from utils.tools import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
PATH = 'experiments/'
EXPERIMENTS = ['server_02', 'server_salinas_03', 'server_indian_pines_02',
               'reduced_01_server_paviau_01', 'reduced_01_server_salinas_01', 'reduced_01_server_indian_pines_01',
               'reduced_05_server_paviau_02', 'reduced_05_server_salinas_01', 'reduced_05_server_indian_pines_01',
               'reduced_10_server_paviau_01', 'reduced_10_server_salinas_01', 'reduced_10_server_indian_pines_01']


NOISES = [['salt_and_pepper', 0], ['salt_and_pepper', 0.001], ['salt_and_pepper', 0.005], ['salt_and_pepper', 0.01],
          ['salt_and_pepper', 0.05],
          ['additive_gaussian', 0.05], ['additive_gaussian', 0.1], ['additive_gaussian', 0.3],
          ['additive_gaussian', 0.5], ['additive_gaussian', 1.0],
          ['multiplicative_gaussian', 0.1], ['multiplicative_gaussian', 0.3], ['multiplicative_gaussian', 0.5],
          ['multiplicative_gaussian', 1.0],
          ['section_mul_gaussian', 1], ['section_mul_gaussian', 2], ['section_mul_gaussian', 3],
          ['section_mul_gaussian', 4], ['section_mul_gaussian', 5], ['section_mul_gaussian', 6],
          ['section_mul_gaussian', 7], ['section_mul_gaussian', 8],
          ['single_section_gaussian', 1], ['single_section_gaussian', 2], ['single_section_gaussian', 3],
          ['single_section_gaussian', 4], ['single_section_gaussian', 5], ['single_section_gaussian', 6],
          ['single_section_gaussian', 7], ['single_section_gaussian', 8]]


# Test SAE-3DDRN runs
def test(config_file):
    cfg = SAE3DConfig(config_file, test=True)

    # Set string modifier if testing best models
    test_best = 'best_' if cfg.test_best_models else ''
    if cfg.test_best_models:
        print('Testing best models from each run!')

    # Load processed dataset
    data = torch.load(cfg.exec_folder + 'proc_image.pth')

    for run in range(cfg.num_runs):
        print(f'TESTING RUN {run + 1}/{cfg.num_runs}')

        # Get normalization type
        sae_data = torch.load(cfg.exec_folder + f'runs/encoded_image_{run}.pth')
        if np.abs(1.0 - np.max(sae_data)) < 1e-5:
            normalization = 'minmax'
        elif np.abs(np.mean(sae_data)) < 1e-5:
            normalization = 'standard'
        else:
            raise ValueError(f'Data normalization not found. Mean: {np.mean(sae_data)}. Max: {np.max(sae_data)}')
        print(f'Normalization: {normalization}')

        _, test_gt, _ = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)
        test_gt = HSIData.remove_negative_gt(test_gt)
        num_classes = len(np.unique(test_gt)) - 1

        # Load autoencoder model
        sae_file = cfg.exec_folder + f'runs/sae_model_run_' + str(run) + '.pth'
        sae = SAE(data.shape[2], cfg.sae_hidden_layers)
        sae.load_state_dict(torch.load(sae_file, map_location=device))
        sae.eval()

        for noise in NOISES:
            print(f'Using {noise[0]} noise with parameter: {noise[1]}')
            noisy_data = add_noise(data, noise)
            sae_noisy_data = sae(torch.Tensor(noisy_data))
            sae_noisy_data = sae_noisy_data.detach().numpy()
            sae_noisy_data, _ = HSIData.normalize(sae_noisy_data, normalization=normalization)

            # Load test ground truth and initialize test loader
            test_dataset = DRNDataset(sae_noisy_data, test_gt, cfg.sample_size, data_augmentation=False)
            test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

            # Load model
            model_file = cfg.exec_folder + f'runs/drn_{test_best}model_run_' + str(run) + '.pth'
            model = nn.DataParallel(DRN(num_classes))
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()

            # Set model to device
            model = model.to(device)

            # Test model from the current run
            report = test_model(model, test_loader)
            save_noise_results(cfg.results_folder, noise, report)


# Main for running test independently
def main():
    # Load config data from training
    for experiment in EXPERIMENTS:
        config_file = PATH + experiment + '/config.yaml'

        test(config_file)


if __name__ == '__main__':
    main()
