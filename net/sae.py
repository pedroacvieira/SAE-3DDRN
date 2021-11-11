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
        self.encoders = []
        self.decoders = []

        # Fill lists according to channel settings
        previous_channels = input_channels
        for num_channels in hidden_channels:
            self.encoders.append(nn.Linear(previous_channels, num_channels))
            self.decoders.append(nn.Linear(num_channels, previous_channels))
            previous_channels = num_channels
        self.decoders.reverse()

        self.training_layer = -1

    def set_training_layer(self, layer):
        self.training_layer = layer
        for i in range(len(self.encoders)):
            if i == layer:
                self.encoders[i].requires_grad_(True)
                self.decoders[i].requires_grad_(True)
            else:
                self.encoders[i].requires_grad_(False)
                self.decoders[i].requires_grad_(False)

    def forward(self, x):
        data = x
        out = x

        if self.training:
            assert self.training_layer >= 0, "A training layer must be selected"

            for i in range(self.training_layer + 1):
                if i == self.training_layer:
                    data = out  # Save input for training layer
                out = self.encoders[i](out)

            out = self.decoders[self.training_layer](out)
        else:
            # If not training, run the entire network
            for encoder in self.encoders:
                out = encoder(out)

        return out, data
