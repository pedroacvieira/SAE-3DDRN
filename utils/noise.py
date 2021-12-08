#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 7 13:31 2021

@author: Pedro Vieira
@description: Implements a function to add noise to a test hyperspectral image
"""

import torch
import numpy as np
import cv2 as cv

# Set image file path
PATH = '../experiments/prototype/'
IMAGE_NAME = 'proc_image.pth'

# Set noise parameters
NOISE_TYPE = 'salt_and_pepper'
SIGNAL_NOISE_RATIO = 0.9


def add_noise(img, noise_amount, noise_type='salt_and_pepper'):
    out = np.copy(img)
    pixels_per_band = img.shape[0] * img.shape[1]

    if noise_type is 'salt_and_pepper':
        # Calculate the number of image salt and pepper noise points
        num_points = int(pixels_per_band * noise_amount)

        for band in range(img.shape[2]):
            for i in range(num_points):
                rand_x = np.random.randint(img.shape[0])  # Generate random x coordinate
                rand_y = np.random.randint(img.shape[1])  # Generate random y coordinate

                if np.random.random() <= 0.5:
                    out[rand_x, rand_y, band] = 0
                else:
                    out[rand_x, rand_y, band] = 1
    elif noise_type is 'gaussian':
        print('Implement it')
    else:
        raise Exception('noise type not implemented')

    return out


# Main for running script independently
def main():
    img = torch.load(PATH + IMAGE_NAME)

    noisy_img = add_noise(img, SIGNAL_NOISE_RATIO, NOISE_TYPE)
    torch.save(noisy_img, PATH + 'img_noise_' + NOISE_TYPE + '.pth')


if __name__ == '__main__':
    main()
