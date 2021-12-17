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
MAX = 1.0


def add_noise(img, noise_param, noise_type='salt_and_pepper'):
    out = np.copy(img)
    pixels_per_band = img.shape[0] * img.shape[1]

    # Applies a salt and pepper noise to each band for a noise_param ratio of the pixels
    # noise_param = fraction of noisy pixels [0.1)
    if noise_type == 'salt_and_pepper':
        # Calculate the number of image salt and pepper noise points
        num_points = int(np.floor(pixels_per_band * noise_param))

        for band in range(img.shape[2]):
            for i in range(num_points):
                rand_x = np.random.randint(img.shape[0])  # Generate random x coordinate
                rand_y = np.random.randint(img.shape[1])  # Generate random y coordinate

                out[rand_x, rand_y, band] = 0 if np.random.random() <= 0.5 else MAX

    # Applies an additive gaussian noise to every pixel with mean and variance defined by noise_param
    # noise_param = sigma; normal = [mu, sigma]
    elif noise_type == 'additive_gaussian':
        for idx in range(out.shape[2]):
            noise = np.random.normal(0.0, noise_param, size=(img.shape[0], img.shape[1]))
            out[:, :, idx] += noise

    # Applies a multiplicative gaussian noise to every pixel with mean and variance defined by noise_param
    # noise_param = sigma; noise = normal(1.0, noise_param)
    elif noise_type == 'multiplicative_gaussian':
        for idx in range(out.shape[2]):
            noise = np.random.normal(1.0, noise_param, size=(img.shape[0], img.shape[1]))
            out[:, :, idx] = np.multiply(out[:, :, idx], noise)
    else:
        raise Exception('Noise type not implemented')

    # Prune out of bounds values
    out_of_bounds_max = np.nonzero(out > MAX)
    out[out_of_bounds_max] = MAX
    out_of_bounds_min = np.nonzero(out < 0)
    out[out_of_bounds_min] = 0

    return out


# Main for running script independently
def main():
    img = torch.load(PATH + IMAGE_NAME)

    noisy_img = add_noise(img, SIGNAL_NOISE_RATIO, NOISE_TYPE)
    torch.save(noisy_img, PATH + 'img_noise_' + NOISE_TYPE + '.pth')


if __name__ == '__main__':
    main()
