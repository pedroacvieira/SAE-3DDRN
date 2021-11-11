# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:48 2021

@author: Pedro Vieira
@description: Implements the training function for the stacked autoencoder
"""

from utils.dataset import SAEDataset
from net.sae import SAE


def train_stacked_autoencoder(data, cfg):
    print('STARTING TRAIN OF STACKED AUTOENCODER')

    dataset = SAEDataset(data.image)

    model = SAE(104, [80, 60, 40, 10])
    return model
