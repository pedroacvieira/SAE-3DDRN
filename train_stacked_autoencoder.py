# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:48 2021

@author: Pedro Vieira
@description: Implements the training function for the stacked autoencoder
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.sae import SAE
from test import test_sae_model
from utils.dataset import SAEDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_stacked_autoencoder(data, train_gt, val_gt, cfg):
    print('STARTING TRAIN OF STACKED AUTOENCODER')

    # Create dataset objects
    train_dataset = SAEDataset(data.image, train_gt)
    val_dataset = SAEDataset(data.image, val_gt)

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.sae_train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.sae_test_batch_size, shuffle=False)

    # Setup model, optimizer, loss and scheduler
    model = SAE(data.image.shape[2], cfg.sae_hidden_layers)
    criterion = nn.MSELoss()

    # Enable GPU training
    model = model.to(device)
    criterion = criterion.to(device)

    # Run epochs
    total_steps = len(train_loader)
    total_layers = len(cfg.sae_hidden_layers)
    for layer in range(total_layers):
        print(f'TRAINING AUTOENCODER LAYER {layer + 1}/{total_layers}')
        model.set_training_layer(layer)

        # Restart optimizer and lr scheduler for every layer
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.sae_learning_rate[layer], betas=cfg.sae_betas,
                                     weight_decay=cfg.sae_weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.sae_scheduler_step[layer],
                                                       gamma=cfg.sae_gamma)

        # Save best models per run
        best_layer = (None, None)
        best_error = float('inf')

        for epoch in range(cfg.sae_num_epochs[layer]):
            print(f'STARTING AUTOENCODER EPOCH {epoch + 1}/{cfg.sae_num_epochs[layer]}')

            running_loss = 0.0

            # Run iterations
            for i, pixels in tqdm(enumerate(train_loader), total=len(train_loader)):
                pixels = pixels.to(device)

                # Forward pass
                outputs = model(pixels)
                loss = criterion(outputs, pixels)
                running_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                if (i + 1) % cfg.sae_print_frequency == 0:
                    avg_loss = running_loss / cfg.sae_print_frequency
                    running_loss = 0.0

                    # Print data
                    tqdm.write(
                        f'\tLayer [{layer + 1}/{total_layers}]\tEpoch [{epoch + 1}/{cfg.sae_num_epochs[layer]}]'
                        f'\tStep [{i + 1}/{total_steps}]\tLoss: {avg_loss:.5f}')

            # Run validation
            if cfg.val_split > 0:
                print(f'STARTING VALIDATION {epoch + 1}/{cfg.sae_num_epochs[layer]}')
                model.eval()
                error = test_sae_model(model, val_loader)
                model.train()

                if best_error > error:
                    best_error = error
                    best_layer = (model.encoders[layer].state_dict(), model.decoders[layer].state_dict())

        # Set the layer values to the best result during training
        print(f'FINISHED TRAINING LAYER {layer + 1}/{total_layers}, LOADING BEST VALUES.')
        model.encoders[layer].load_state_dict(best_layer[0])
        model.decoders[layer].load_state_dict(best_layer[1])

    return model.cpu()
