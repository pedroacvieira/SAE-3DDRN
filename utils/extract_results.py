#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

############
# Set file #
############
PATH = '../../../Results/'
EXPERIMENT = 'full/'
FILE = 'test_paviau_dffn.txt'
# DATASETS = ['paviau', 'indian_pines', 'salinas']  # Use this for all datasets
DATASETS = ['paviau']
NETWORKS = ['sdmm', 'dffn', 'vscnn', 'sae3ddrn']

VALUE_POSITION = 3
NUM_RUNS = 10
PRINT_PER_CLASS_ACCURACY = True


# Get test results from text file
def get_values(filename):
    overall_accuracy = []
    average_accuracy = []
    kappa_coefficients = []
    per_class_accuracy = []

    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            # Check for OA
            if 'OVERALL ACCURACY' in line:
                words = line.split(' ')
                overall_accuracy.append(float(words[VALUE_POSITION]))
            # Check for AA
            elif 'AVERAGE ACCURACY' in line:
                words = line.split(' ')
                average_accuracy.append(float(words[VALUE_POSITION]))
            # Check for kappa
            elif 'KAPPA COEFFICIENT' in line:
                words = line.split(' ')
                kappa_coefficients.append(float(words[VALUE_POSITION]))
            elif 'PER CLASS ACCURACY' in line:
                run_class_acc = []
                line = file.readline()
                while line != '\n':
                    # line = line[1:] if line[0] == ' ' else line  # Remove space in the beginning of a line
                    run_class_acc.extend(line.strip('[]\n').split(' '))
                    line = file.readline()
                non_empty = [float(x) for x in run_class_acc if x]
                per_class_accuracy.append([float(x) for x in run_class_acc if x])

            # Get next line
            line = file.readline()

    assert len(overall_accuracy) == NUM_RUNS, f'File should have {NUM_RUNS} runs! [1]'
    assert len(average_accuracy) == NUM_RUNS, f'File should have {NUM_RUNS} runs! [2]'
    assert len(kappa_coefficients) == NUM_RUNS, f'File should have {NUM_RUNS} runs! [3]'
    assert len(per_class_accuracy) == NUM_RUNS, f'File should have {NUM_RUNS} runs! [4]'

    oa = np.array(overall_accuracy)
    aa = np.array(average_accuracy)
    kappa = np.array(kappa_coefficients)
    class_acc = np.array(per_class_accuracy)
    return oa, aa, kappa, class_acc


# Main for running script independently
def main():
    for data in DATASETS:
        for net in NETWORKS:
            file = 'test_' + data + '_' + net + '.txt'
            filename = PATH + EXPERIMENT + file
            oa, aa, kappa, class_acc = get_values(filename)

            print(f'TEST: {net} with {data}')
            print('#' * 15)
            print(f'OA: {oa.mean():.6f} ({oa.std():.6f})')
            print(f'AA: {aa.mean():.6f} ({aa.std():.6f})')
            print(f'Kappa: {kappa.mean():.6f} ({kappa.std():.6f})')
            print('-' * 15)
            print(f'Max OA: {np.max(oa):.5f}')
            print(f'Min OA: {np.min(oa):.5f}')
            if PRINT_PER_CLASS_ACCURACY:
                print('-' * 15)
                print(f'Per Class Accuracy: {np.mean(class_acc, axis=0)}')
            print('')


if __name__ == '__main__':
    main()
