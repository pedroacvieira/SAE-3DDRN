# SAE-3DDRN
This is an implementation of the SAE-3DDRN network presented in "A combination method of stacked autoencoder and 3D deep residual network for hyperspectral image classification".
This implementation is purely based on the paper without any access to the originally implemented network.

# Tested with
* Python 3.8 and 3.9
* NumPy 1.21.2
* Pytorch 1.9.0
* SciPy 1.7.1
* MatPlotLib 3.4.3
* Tqdm 4.62.1
* Using both CPU and GPU

# Run the SAE-3DDRN
Please set your parameters in `config.yaml`, `train.py` or `test.py` before running them.
More importantly, every experiment needs to have a different name.
For more information, see section "Config file".

To train, run:
```bash
# Trains network multiple times (see parameters in file)
python3 train.py
``` 

To test, run:
```bash
# Tests all runs saved in a given directory
python3 test.py
```

# About datasets
To download the datasets, please use the script `./download_datasets.sh`.
These datasets and others can be found [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).
The used datasets for this implementation are: PaviaU, Indian Pines and Salinas.

# File structure

The `./datasets` directory stores all datasets once they have been downloaded.
All experiment-related data is stored in `./experiments` once a train or test is run.
The `./net` directory has the architectures of the networks and their building blocks.
Lastly, the `./utils` directory contains multiple other implementations used in the training and testing of the networks, as well as IO functions.
It also contains scripts related to the tests, as described below.

In `./utils`, the script `model_from_checkpoint.py` can be used to extract a network model file (with all network parameters) from one of the checkpoints stored for a specific experiment in `./experiments/<some_experiment>/checkpoints/`.
The script `dataset_histograms.py` was used to generate the histograms for all datasets in Chapter 6 of the master thesis.
The remaining two scripts, `extract_results.py` and `extract_noise_results.py`, use the test data stored in `./experiments/<some_experiment>/results/` to generate the table data and the figures displayed in the master thesis.

# Config file

This file is divided in sections related to the dataset, network and additional options.

## Dataset Section
Contains the parameters related to the dataset, train-test split, and the experiment.

### `dataset` (string)
* Desired dataset. Can be `PaviaU`, `IndianPines` or `Salinas`
### `experiment` (string)
* Name for the experiment, which will be used to save load all information
### `data_folder` (string)
* Default value: './datasets/'
* Directory where the desired dataset is kept
### `exec_folder` (string)
* Default value: './experiments/'
* Directory where to keep all the experiment data
### `split_folder` (string)
* Default value: 'data_split/'
* Where to store dataset splits
### `val_split` (float)
* Fraction from the dataset used for validation [0, 1]
### `train_split` (float)
* Fraction from the dataset used for training [0, 1]
### `generate_samples` (boolean)
* Whether the samples should be generated (False to load previously saved samples)
### `max_samples` (integer or null)
* Maximum amount of training samples per class (null for no limit)

The most important things to be edited are the dataset name, experiment name, and the train-val-test split.
The other options are simply directory locations, which can stay with their default values.

## Network Section
The second section contains the network's hyperparameter.

### `train_batch_size` (integer)
* Default value: 128
* Batch size for every train iteration
### `test_batch_size` (integer)
* Default value: 64
* Batch size for every test iteration
### `sample_size` (integer)
* Default value: 25
*Window size for every sample/pixel input
### `num_runs` (integer)
* Default value: 10
* The amount of time the whole experiment should run
### `num_epochs` (integer)
* Default value: 50
* Number of epochs per run
### `learning_rate` (float)
* Default value: 0.01
* Initial learning rate
### `betas` (float list)
* Default value: [0.9, 0.999]
* Betas for Adam optimizer
### `weight_decay` (float)
* Default value: 1e-4
* The weight decay for the optimizer
### `gamma` (float)
* Default value: 0.1
* Gamma parameter for the lr scheduler
### `drop_out` (float)
* Default value: 0.4
* Drop out parameter for the 3DDRN network

## Stacked Autoencoder Section
This section has the network hyperparameters for the SAE. The list parameters can have different sizes. The size of these lists is the number of layers of the SAE and every item in these lists refers to the parameters of a specific layer. Therefore, all list parameters (except betas) need to have the same size.

### `sae_hidden_layers` (integer list)
* Default values: [80, 60, 40, 10]
* Number of channels for each of the hidden layers of the SAE
### `sae_train_batch_size` (integer)
* Default values: 200
* Batch size for every train iteration
### `sae_test_batch_size` (integer)
* Default values: 50
* Batch size for every test iteration
### `sae_num_epochs` (integer list)
* Default values: [15, 12, 10, 8]
* Number of epochs for every autoencoder
### `sae_learning_rate` (float list)
* Default values: [0.01, 0.01, 0.01, 0.01]
* Initial learning rate for every autoencoder
### `sae_betas` (float list)
* Default values: [0.9, 0.999]
* Betas of Adam optimizer for the stacked autoencoder
### `sae_weight_decay` (float)
* Default values: 1e-4
* The weight decay for the optimizer for the stacked autoencoder
### `sae_gamma` (float)
* Default values: 0.1
* Gamma parameter for the lr scheduler for the stacked autoencoder
### `sae_scheduler_step` (integer list)
* Default values: [1000, 600, 500, 300]
* Step size for the lr scheduler for every autoencoder

## Extra Section
The last section contains extra options, as shown below

### `test_best_models` (boolean)
* Default value: True
* Whether to test the best model of each run
### `use_checkpoint` (boolean)
* Default value: False
* Whether to load a checkpoint during training
### `results_folder` (string)
* Default value: 'results/'
* Folder where to write the validation and test results
### `checkpoint_folder` (string)
* Default value: 'checkpoints/'
* Folder where to keep checkpoints
### `checkpoint_file` (string or null)
* Default value: null
* What checkpoint file to load (null for the latest)
### `delete_checkpoints` (boolean)
* Default value: True
* Delete checkpoints after successfully training the network (saves storage)
### `print_frequency` (integer)
* Default value: 40
* The amount of iterations between every step/loss print

Once every parameter has been setup, the train or test can be started with the lines provides above.

# Obtaining test results
Once the train is complete, the models that obtained the best validation accuracies will be stored in its experiment directory `./experiments/<experiment_name>/runs/`.
After this directory has successfully been initialized after a train, its models can be tested with the line presented above.
The test scrip tests the last experiment by default, but this can be changed by editing the variable `CONFIG_FILE` in `./test.py`.

When an experiment is tested, all the results are stored in the directory `./experiments/<experiment_name>/results/`.
The results can be directly accessed in this location, or obtained using the script `./utils/extract_results.py`, which will calculate the average and variance of the results for all trained models in this experiment.

The script `noise_test.py` is made to use multiple experiments and test them with different amounts of noise.
For that, it uses two global variables, `EXPERIMENTS` and `NOISES`, which are lists containing the names of the experiments and the types and amounts of noise, respectively.
After these two variables have been correctly set to the desired experiments and noises, the script can be run with
```bash
# Tests all experiments provided in 'EXPERIMENTS' with all noises in 'NOISES'
python3 noise_test.py
```
After this, the figures with the results of these noise tests can be generated with the script `./utils/extract_noise_results.py`.
In this script, the desired datasets, networks, test cases, and noise types must be specified.