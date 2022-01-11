#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:28 2021

@author: Pedro Vieira
@description: Implements network architecture of the 3D DRN for the SAE-3DDRN
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()

        # Shortcut parameters
        self.batch_norm = nn.BatchNorm3d(input_channels)
        self.relu = nn.ReLU()

        # Block parameters
        self.block = nn.Sequential(nn.Conv3d(input_channels, input_channels, (3, 3, 3), padding=1),
                                   nn.BatchNorm3d(input_channels), self.relu,
                                   nn.Conv3d(input_channels, input_channels, (3, 3, 3), padding=1),
                                   nn.BatchNorm3d(input_channels), self.relu,
                                   nn.Conv3d(input_channels, input_channels, (3, 3, 3), padding=1),
                                   nn.BatchNorm3d(input_channels))

    def forward(self, x):
        out = self.block(x)
        out += self.relu(self.batch_norm(x))

        return out


class DRN(nn.Module):

    def __init__(self, num_classes, drop_out=0.4):
        super(DRN, self).__init__()
        # Input size = [1, 10, 25, 25]
        self.relu = nn.ReLU()

        self.block1 = nn.Sequential(nn.Conv3d(1, 16, (3, 3, 3)),  # Size = [16, 8, 23, 23]
                                    nn.BatchNorm3d(16), self.relu,
                                    ResidualBlock(16), self.relu,
                                    nn.MaxPool3d((1, 2, 2)))  # Size = [16, 8, 11, 11]
        self.block2 = nn.Sequential(nn.Conv3d(16, 32, (3, 3, 3), padding=1),
                                    nn.BatchNorm3d(32), self.relu,
                                    ResidualBlock(32), self.relu,
                                    nn.MaxPool3d((1, 2, 2)))  # Size = [32, 8, 5, 5]
        self.block3 = nn.Sequential(nn.Conv3d(32, 64, (3, 3, 3), padding=1),
                                    nn.BatchNorm3d(64), self.relu,
                                    ResidualBlock(64), self.relu,
                                    nn.MaxPool3d((2, 2, 2)),
                                    nn.BatchNorm3d(64))  # Output size should be [64, 4, 2, 2]
        self.classifier = nn.Sequential(nn.Linear(1024, 256), self.relu, nn.Dropout3d(drop_out),
                                        nn.Linear(256, 128), self.relu, nn.Dropout3d(drop_out),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = out.view(-1, 1024)
        out = self.classifier(out)
        return out
