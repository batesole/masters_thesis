# Master's Thesis Repository of A. Batesole

**As of 16 December 2025 this repository is still under construction.**

This is the public repository of the work related to my master's thesis.  The main work of the thesis was to replace using a single, static activation function throughout a network with a linear weighted blend of functions, called an Adaptive Blending Unit (ABU). 

There are four python files included here:
- **resnet_baseline.py:** Program to train a baseline ResNet using one activation function throughout the network
- **resnet_ABU.py:** Program to train a baseline ResNet replacing the single activation function with the ABU
- **alpha_analysis.py:** Program to analyze the resulting alpha weights from the ABU
- **layer_func.py:** Program that saves plots of the resulting layer functions from the ABU

A link to my thesis will be included here when it is published.


## Setup
These programs were run in a conda environment, which was exported to an environment.yml file included here.  If one wishes to run these programs in the same environment, all that is needed is to create an environment based on the environment.yml:

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


## Global Parameters of ResNet Programs
In the two ResNet python files the global parameters that can be modified are at the top of the program underneath the imports.  These parameters include:
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
- **num_runs:** How many times to train and test the model.  Each run uses a randomly generated seed.


## File Outputs
Each program saves some form of files during runtime.  They all begin with looking at the current working directory.

resnet_baseline.py and resnet_ABU.py will save results into a folder named "run_i", where i is the run number, and a folder named "eval_results."  The folder eval_results contains the model.evaluate results for every run, saved as a .npy file.  The folder run_i saves results from each run, including accuracy plots, loss plots, and training history.  For resnet_ABU.py, run_i also includes a folder named alpha_logs, which saves the alpha values over all epochs for each function, and each layer.  The values are plotted across epochs, and also saved into .csv files.

layer_func.py plots the resulting function of each layer, based on the layer .csv files in the current directory, and saves each layer's function as a .png.  The file names are hard coded, based on how they get saved from resnet_ABU.py.

alpha_analysis.py does some analysis of the saved alpha values from resnet_ABU.py.  It looks for a parent directory (the default name is "data") within the current working directory, where it expects to see an experiment name, which contains all the saved folders from resnet_ABU.py (where each run is saved into a separate run_i folder).  It goes through every experiment and run folder and reads in all alpha.csv files (including layerx_alphas.csv and funcx_alphas.csv).  The visual below shows the expected hierarchy:

```
- PARENT DIRECTORY - 
    |- Experiment 1
    |  |- Run 1
    |  |  |- alpha_logs
    |  |  |  |- layer1_alphas.csv
    |  |  |  |- func1_alphas.csv
    |  |  |- ...
    |  |- ...
    |  |- Run n
    |  |  |- alpha_logs
    |  |  |  |- alpha csv files
    |- Experiment ...
    |  |- run ...
    |  |  |- alpha_logs
    |  |  |  |- alpha csv... 
```
After reading in all the alpha .csv files, analysis is done based on the final alpha values per function and per layer.  Plots of the final alphas per function and scatterplots of the final alphas per layer are saved in the current directory as .png files.  Summary statistics are also done but are only saved to a variable.

## Alpha Analysis Figures
Each run from resnet_ABU.py saves the resulting alpha weights from the blended functions.  Results include alpha per layer of each function, alpha per function of each layer, and model accuracy.  Results are stored first by experiment (labeled by function pool), then by run number. 

Some examples will be uploaded here, along with some basic analysis of the alpha values, including the final resulting function per layer, and plots of the alpha weights.
