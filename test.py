#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 17:57 2021

@author: Pedro Vieira
@description: Implements the test function for the SAE-3DDRN network
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from utils.config import SAE3DConfig
from utils.dataset import SAEDataset, DRNDataset
from utils.tools import *
from net.sae import SAE
from net.drn import DRN

# Import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
CONFIG_FILE = ''  # Empty string to load default 'config.yaml'


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
        num_classes = len(np.unique(test_gt)) - 2

        test_dataset = SAEDataset(data, test_gt)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

        # Load model
        sae_file = cfg.exec_folder + f'runs/sae_model_run_' + str(run) + '.pth'
        sae = SAE(data.shape[2], cfg.sae_hidden_layers)
        sae.load_state_dict(torch.load(sae_file))
        sae.eval()

        # Set model to device
        sae = sae.to(device)

        sae_report = test_sae_model(sae, test_loader, num_classes + 1)

        sae_data = torch.load(cfg.exec_folder + f'runs/encoded_image_{run}.pth')

        # Remove undefined class from ground truth
        test_gt = HSIData.remove_negative_gt(test_gt)

        print(f'TESTING SAE-3DDRN MODEL FROM RUN {run + 1}/{cfg.num_runs}')
        # Load test ground truth and initialize test loader
        test_dataset = DRNDataset(sae_data, test_gt, cfg.sample_size, data_augmentation=False)
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
        filename = cfg.results_folder + 'test.txt'
        save_results(filename, report, sae_report, run)

    if cfg.use_tensorboard:
        writer.close()


# Test stacked autoencoder
def test_sae_model(model, loader, num_classes):
    total_loss = 0.0
    per_class_loss = np.array([0.0] * num_classes)
    counter = np.array([0] * num_classes, dtype=int)
    model.test(True)
    with torch.no_grad():
        for pixels, labels in tqdm(loader, total=len(loader)):
            pixels = pixels.to(device)
            outputs = model(pixels)

            for i, label in enumerate(labels):
                label = label.item()
                per_class_loss[label] += f.mse_loss(outputs[i], pixels[i])
                total_loss += f.mse_loss(outputs[i], pixels[i])
                counter[label] += 1
    model.test(False)

    avg_loss = total_loss / np.sum(counter)
    per_class_loss = per_class_loss / counter

    print(f'- Average loss: {avg_loss:.5f}')
    print(f'- Per class loss: {per_class_loss}')

    report = {
        'avg_loss': avg_loss,
        'per_class_loss': per_class_loss
    }

    return report


# Function for performing the tests for a given model and data loader
def test_model(model, loader, writer=None):
    labels_pr = []
    prediction_pr = []
    with torch.no_grad():
        total_predicted = np.array([], dtype=int)
        total_labels = np.array([], dtype=int)
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            # for images, labels in loader:
            # Get input and compute model output
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Get predicted outputs
            _, predicted = torch.max(outputs, 1)

            # Save total values for analysis
            total_predicted = np.append(total_predicted, predicted.cpu().numpy())
            total_labels = np.append(total_labels, labels.cpu().numpy())

        report = get_report(total_predicted, total_labels)
        print(f'- Overall accuracy: {report["overall_accuracy"]:f}')
        print(f'- Average accuracy: {report["average_accuracy"]:f}')
        print(f'- Kappa coefficient: {report["kappa"]:f}')

        if writer is not None:
            # Accuracy per class
            classes = range(9)
            for i in classes:
                labels_i = labels_pr == i
                prediction_i = prediction_pr[:, i]
                writer.add_pr_curve(str(i), labels_i, prediction_i, global_step=0)

    return report


# Compute OA, AA and kappa from the results
def get_report(y_pr, y_gt):
    classify_report = metrics.classification_report(y_gt, y_pr)
    confusion_matrix = metrics.confusion_matrix(y_gt, y_pr)
    class_accuracy = metrics.precision_score(y_gt, y_pr, average=None)
    overall_accuracy = metrics.accuracy_score(y_gt, y_pr)
    average_accuracy = np.mean(class_accuracy)
    kappa_coefficient = kappa(confusion_matrix)

    # Save report values
    report = {
        'classify_report': classify_report,
        'confusion_matrix': confusion_matrix,
        'class_accuracy': class_accuracy,
        'overall_accuracy': overall_accuracy,
        'average_accuracy': average_accuracy,
        'kappa': kappa_coefficient
    }
    return report


# Compute kappa coefficient
def kappa(confusion_matrix):
    data_mat = np.mat(confusion_matrix)
    p_0 = 0.0
    for i in range(confusion_matrix.shape[0]):
        p_0 += data_mat[i, i] * 1.0
    x_sum = np.sum(data_mat, axis=1)
    y_sum = np.sum(data_mat, axis=0)
    p_e = float(y_sum * x_sum) / np.sum(data_mat)**2
    oa = float(p_0 / np.sum(data_mat) * 1.0)
    cohens_coefficient = float((oa - p_e) / (1 - p_e))
    return cohens_coefficient


# Main for running test independently
def main():
    test()


if __name__ == '__main__':
    main()
