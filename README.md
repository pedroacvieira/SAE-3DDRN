# SAE-3DDRN
This is an implementation of the SAE-3DDRN network presented in "A combination method of stacked autoencoder and 3D deep residual network for hyperspectral image classification".
This implementation is purely based on the paper without any access to the originally implemented network.

# Tested with
Python 3.8 and 3.9

Pytorch 1.9.0  

CPU and GPU

# Run the SAE-3DDRN
Please set your parameters in train.py or test.py before running them. 

To train, run:
```bash
# Trains network multiple times (see parameters in file)
python train.py
``` 

To test, run:
```bash
# Tests all runs saved in a given directory
python test.py
```

# About datasets
The datasets are available [here](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).
The used datasets for this implementation are: PaviaU, Indian Pines and Salinas.

# Config file
Please refer to the config file `config.yaml` for details about the possible configurations of the network/training/testing.

