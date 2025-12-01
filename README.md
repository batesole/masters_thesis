# Master's Thesis Repository of A. Batesole

**As of 1 December 2025 this repository is still under construction.**

This is the public repository of the work related to my master's thesis.  The main work of the thesis was to replace using a single, static activation function throughout a network with a linear weighted blend of functions, called an Adaptive Blending Unit (ABU). 

There are four python files included here:
- **resnet_baseline.py** 
- **resnet_ABU.py**
- **alpha_analysis.py**
- **layer_func.py**





## Setup
This program was run in a conda environment, which was exported to an environment.yml file included here.  If one wishes to run these programs in the same environment, all that is needed is to use 

```
# creating an env from a yaml
conda env create -f environment.yml
```

The biggest problems tend to arise from the package versions of Cuda, Cudnn, Tensorflow, Keras, Numpy, and Python.  The versions of those packages are:
- Cuda Version: 12.5.1
- Cudnn version: 9
- Tensorflow version:  2.18.0
- Numpy version:  2.0.2
- Keras version:  3.8.0
- Python version: 3.11.13


## Global Parameters
In the two ResNet python files the global parameters that can be modified are at the top of the program.  These parameters include:
- **num_models:** Included only in resnet_baseline.py.  The file is set up to run either one model, or all eight models.  If you wish to run just one model, set this variable to 1, otherwise set it to 8.  (This is only included in resnet_baseline.py)
- **lr_start:** This is the starting learning rate.  The learning rate scheduler is written so that at epoch1 it divides lr_start by 10, and at epoch2 it divides lr_start by 100.
- **epoch1:** This is the epoch where lr_start will thereafter be divided by 10.
- **epoch2:** This is the epoch where lr_start will thereafter be divided by 100.
- **epochs:** This is the total number of epochs the model will be trained for.
- **batch_size:** This is the number of samples per batch of computation when fitting the model.  Recommended to use 128 for Cifar.
- **num_blocks:** When using the main architecture for ResNet, this determines the depth.  Above this variable are variables named resXX for each size of ResNet, for example res18 creates the ResNet18 architecture.  Set num_blocks to the variable of the desired size, for example num_blocks=res18 will train the ResNet18 architecture.
- **block_type:** There are two block types, bottleneck and basic, and *one* of them needs to be uncommented.  Use bottleneck block type for ResNet50, ResNet101, and ResNet152, and use basic block type for all others.
- **net_depth:** This is used for generating plot titles.  Write the numerical value of the network depth you are using.
- **depth:** This is the ResNet size when using the Cifar variant of the ResNet architecture.  Enter a numerical value of the desired depth.  The value must be 6n+2, for example 20, 32, 44, etc.
- **act_func:** Included only in resnet_baseline.py.  When running a single model, this specifies the activation function to be used in the model.
- **act_funcs:** Included only in resnet_ABU.py.  A list of the activation functions to be included in the blending pool.
- **kernel_init:** This specifies the initializer to be used when initializing the model.

## File Outputs


## Alpha Analysis Figures
Each run saves the resulting alpha weights from the blended functions.  These will be uploaded elsewhere and a link included here for anyone curious to see the resulting blended function weights.

These will be the results from all runs of the various alpha figures.  Results include alpha per layer of each function, alpha per function of each layer, and final functions of each layer.  Results are stored first by function pool, then by run number. 
