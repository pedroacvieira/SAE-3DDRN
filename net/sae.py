#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:37 2021

@author: Pedro Vieira
@description: Implements network architecture of the stacked autoencoder for the SAE-3DDRN
"""

import torch.nn as nn


class SAE(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(SAE, self).__init__()

        # Create list for the encoders and decoders
        self.num_layers = len(hidden_channels)
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        # Fill lists according to channel settings
        previous_channels = input_channels
        for i, num_channels in enumerate(hidden_channels):
            self.encoders.append(nn.Sequential(nn.Linear(previous_channels, num_channels),
                                               nn.Tanh()))
            self.decoders.append(nn.Sequential(nn.Linear(num_channels, previous_channels),
                                               nn.Sigmoid() if i == 0 else nn.Tanh()))
            previous_channels = num_channels

        self.training_layer = -1
        self.testing = False

    def test(self, test_status):
        self.testing = test_status

    def train(self, mode=True):
        super(SAE, self).train(mode)

        if mode and self.training_layer >= 0:
            self.set_training_layer(self.training_layer)

    def set_training_layer(self, layer):
        self.training_layer = layer
        for i in range(len(self.encoders)):
            if i == layer:
                self.encoders[i].train()
                self.decoders[i].train()
            else:
                self.encoders[i].eval()
                self.decoders[i].eval()

    # It outputs decoded values for training and testing modes and encoded values otherwise
    def forward(self, x):
        assert self.training_layer >= 0 or not self.training, "A training layer must be selected for training"

        num_layers = len(self.encoders) if self.training_layer < 0 else self.training_layer + 1
        layers = range(num_layers)

        out = x
        for i in layers:
            out = self.encoders[i](out)

        if self.training or self.testing:
            for i in reversed(layers):
                out = self.decoders[i](out)

        return out
