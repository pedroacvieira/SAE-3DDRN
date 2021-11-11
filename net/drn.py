#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:42 2021

@author: Pedro Vieira
@description: Implements network architecture of the 3D DRN for the SAE-3DDRN
"""

import torch
import torch.nn as nn


class DRN(nn.Module):

    def __init__(self, input_channels, num_classes):
        super(DRN, self).__init__()

    def forward(self, x):
        out = x
        return out
