# -*- coding: utf-8 -*-
"""
ResNet with CIFAR-100

ResNet architecture to be used with CIFAR-100.  Adaptive Blending Units (ABUs) are 
implemented in place of single activation functions.  Program loads the dataset using
tensorflow, so it does not need to be downloaded separately.

This program includes architecture to build ResNet based on both the ImageNet
style architecture and Cifar style architecture.  However, it should be noted that
the final architecture used for experiments used the ImageNet style with the first 
global max pool removed, so that layer is commented out.

The README in the github repository includes an explanation of the global parameters.

This program trains the architecture using a random seed each run.  Each run creates 
its own folder to save history plots and alpha weights from blending the functions 
together, and csv files of the model evaluations. 



"""

import time
import datetime
import numpy as np
import random
import os
import sys
from collections import defaultdict
import tensorflow as tf
import keras
import pandas as pd

# keras imports
from tensorflow.keras import layers, models
from keras import regularizers
from keras import optimizers
from tensorflow.keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler

# for image augmentation
import tensorflow_datasets as tfds

# plots
import matplotlib.pyplot as plt


# setting all seeds for debugging purposes
#seed_value = 1337
# set seeds 
#os.environ['PYTHONHASHSEED'] = str(seed_value)
#random.seed(seed_value)
#np.random.seed(seed_value)
#tf.random.set_seed(seed_value)




#%%  GLOBAL PARAMETERS
# define which activation functions we want in the pool
# use format tf.nn.act_func (such as tf.nn.relu)
#act_funcs = None
act_funcs = [    
    tf.nn.relu,
    tf.nn.leaky_relu, 
    tf.nn.sigmoid,
    tf.nn.tanh,
    tf.nn.silu, 
    tf.nn.elu,
    tf.nn.selu,
    tf.nn.gelu
#    tf.identity
#    tf.nn.softsign
]



# define the kernel initializer
# he_uniform for relu family, 
# glorot_uniform for sigmoid family
# lecun_uniform for elu family
kernel_init = 'lecun_uniform'

### Model related parameters
# learning rate, which epochs to change the learning rate, and weight decay
# lr scheduler defined after model def
lr_start = 0.1
epoch1 = 30
epoch2 = 60
weight_decay = 0.0001
# Define epochs and batch size
epochs = 90
batch_size = 128

# ResNet sizes for use with ImageNet
# UNCOMMENT block_type FOR THE RESPECTIVE SIZE
# resnets 18 and 34 have smaller block types
res10 = [1,1,1,1]
res18 = [2,2,2,2]
res34 = [3,4,6,3]
block_type = 'basic'
# the rest of the resnets have larger block types
res50 = [3,4,6,3]
res101 = [3,4,23,3]
res152 = [3,8,36,3]
#block_type = 'bottleneck'

# select which ResNet model size to use here
# uncomment the corresponding block type above (basic or bottleneck) 
num_blocks = res18
# network depth is used for making the titles of plots
net_depth = 18

# ResNet size for use with Cifar (6n+2 = 20, 32, 44, 56, 110, 1202)
depth = 56

# define the number of times to the model 
# each run will use a random seed
num_runs = 10




#%% CLASS DEFINITION

'''
to make custom layers in tensorflow, create a subclass layer, and then
define __init__, build, and call.

__init__ = initializes the layer.  Can pass arguments in for initializing
	- always call super().__init__(**kwargs) with **kwargs to support keras
	integration
		- **kwargs is keyword arguments.  Keras uses a base layer to track
		and store args, which get passed through **kwargs
	
build = put trainable parameters here

call = transformation from inputs to outputs; the layer's forward pass
	- call automatically runs build the first time it is called.
    
We have added in a method to get the activation function names and return
them as a string 

'''
# define the adaptive blending unit as a custom layer 
class adaptive_blending(layers.Layer):
    
    # option to send pool of activation functions
    def __init__(self, act_funcs=None, **kwargs):
    
        # implicit calling of super() preferred for Python 3+
        super().__init__(**kwargs)
        
        # create base pool of all functions if none are sent 
        if act_funcs==None:
            self.act_funcs = [
                tf.nn.relu,
                tf.nn.leaky_relu,
                tf.nn.sigmoid,
                tf.nn.tanh,
                tf.nn.silu,
                tf.nn.elu,
                tf.nn.selu,
                tf.nn.gelu
            ]
            
        else:
            self.act_funcs = act_funcs
            
        # store how large the pool of functions is
        self.num_funcs = len(self.act_funcs)
    
    
    def build(self, input_shape):
        # define the vector of blending weights
        # shape must be a tuple
        #tf.keras.initializers.Ones()
        alpha_init = 1/self.num_funcs
        self.alpha = self.add_weight(
            shape = (self.num_funcs,),
            initializer = tf.keras.initializers.Constant(alpha_init),
            trainable = True,
            name = "alpha"
        ) 
    
    
    def call(self, inputs):
        # unconstrained blending
        # add constraints to alpha here if desired
        output = 0
        for i, fn in enumerate(self.act_funcs):
            output += self.alpha[i] * fn(inputs)
        return output
    
    
    def get_func_names(self):
        # return the activation functions as string names (for logging)
        return [getattr(f, '__name__', str(f)) for f in self.act_funcs]



'''
We want to log the alpha values that are learned for each function in each layer
for later analysis.  We will save all values during training.  To do this, we will
use tensorflow callbacks.  The methods that can be hooked into during training are
listed here https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback

Since we want to log during training, we will use def on_epoch_end to save the values
each epoch.  We will therefore set up the variable used to store values in 
def on_train_begin, and tie it up in def on_train_end.




'''
# set up a callback to log the alpha values 
class alpha_logger(tf.keras.callbacks.Callback):
    # args = save_to_csv, output_dir
    # **kwargs not needed here
    def __init__(self, save_to_csv=True, save_plots=True, output_dir="alpha_logs"):
        super().__init__()
        self.save_to_csv = save_to_csv
        self.save_plots = save_plots
        self.output_dir = output_dir
        # format will be {layer: [alpha_epoch_1, ... alpha_epoch_n] } 
        # each alpha_epoch is the vector of alpha values
        self.alpha_history = {}
        self.func_names = {}
        
        # check if directory exists
        if self.save_to_csv and os.path.exists(output_dir):
            print(f"\nError: {output_dir} directory exists.  Move or delete files.  Exiting program.")
            sys.exit(1)
        if self.save_to_csv and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    
    # Initialize alpha_history for each function and layer
    def on_train_begin(self, logs=None):
        # for each activation function layer, create a new list for that layer
        for layer in self.model.layers:
            if isinstance(layer, adaptive_blending):
                self.alpha_history[layer.name] = []
                self.alpha_history[layer.name].append(layer.alpha.numpy().copy())
                # also get the activation function names
                self.func_names[layer.name] = layer.get_func_names()
                
                
    # save alpha values during training    
    def on_epoch_end(self, epoch, logs=None):
        # for each activation function layer, save the current alpha values
        for layer in self.model.layers:
            if isinstance(layer, adaptive_blending):
                self.alpha_history[layer.name].append(layer.alpha.numpy().copy())
    
    
    # Save alpha values to csv and plots (if specified) 
    # alpha values get saved for each layer, and separately for each function
    def on_train_end(self, logs=None):
    
        # let user know if alpha values were not saved or plotted
        if not (self.save_to_csv or self.save_plots):
            print("\nLog of alpha values not saved or plotted.")
            return
        
        # create a dictionary to collect history of alphas per function
        func_alpha_hist = {}
        
        # save and plot based on each layer
        for layer_name, history in self.alpha_history.items():
            
            # convert to DataFrame
            func_names = self.func_names[layer_name]
            df = pd.DataFrame(history, columns=func_names)
            df.index.name = 'epoch'
            
            # save to csv format (rather than .npy)
            if self.save_to_csv:
                csv_path = os.path.join(self.output_dir, f"{layer_name}_alphas.csv")
                df.to_csv(csv_path)
                
            # make and save plots
            if self.save_plots:
                plot_path = os.path.join(self.output_dir, f"{layer_name}_alphas_plot.png")
                plt.figure(figsize=(12, 6), constrained_layout=True)
                for func in func_names:
                    plt.plot(df.index, df[func], label=func)
                plt.title(f"Alpha Weights of {layer_name}")
                plt.xlabel("Epoch")
                plt.ylabel("Alpha")
                plt.legend(bbox_to_anchor=(1,1), loc='upper left')
                plt.savefig(plot_path)
                plt.close()
                
            # create the history of alpha based on each function
            for func in func_names:
                if func not in func_alpha_hist:
                    func_alpha_hist[func]={}
                func_alpha_hist[func][layer_name] = df[func].values.tolist()

        
        print("alpha values saved per layer")
        
        # save and plot based on each function
        
        # iterate through each function from our func_alpha_hist
        for func_i, layers_dict in func_alpha_hist.items():
            func_df = pd.DataFrame(layers_dict)
            func_df.index.name = 'epoch'
            
            # save to csv format 
            if self.save_to_csv:
                csv_path = os.path.join(self.output_dir, f"{func_i}_alphas.csv")
                func_df.to_csv(csv_path)
                
            # make and save plots
            if self.save_plots:
                plot_path = os.path.join(self.output_dir, f"{func_i}_alphas_plot.png")
                plt.figure(figsize=(12, 6), constrained_layout=True)
                for layer in func_df.columns:
                    plt.plot(func_df.index, func_df[layer], label=layer)
                plt.title(f"Alpha Weights of {func_i}")
                plt.xlabel("Epoch")
                plt.ylabel("Alpha")
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
                plt.savefig(plot_path)
                plt.close()
                
        print("alpha values saved per function")
            
        # save function mapping
        func_map_path = os.path.join(self.output_dir, "function_mapping.txt")
        with open(func_map_path, "w") as f:
            for layer_name, func_list in self.func_names.items():
                f.write(f"{layer_name}: {', '.join(func_list)}\n")







'''

Checking numerics because we are getting nan for loss sometimes

'''
# pass the message to it so we know what layer we're on
class CheckNumericsLayer(layers.Layer):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def call(self, inputs):
        return tf.debugging.check_numerics(inputs, self.message)






#%% FUNCTION DEFINITIONS

### ResNet models

"""
There are two different base functions to build the architecture, one with
the ImageNet style and one with the Cifar style.  The final architecture for
experiments used a modified version of the ImageNet style, so the architecture
here is not the original ImageNet style architecture.

Below are notes taken from the paper of ResNet.


(This first section of notes was related to Resnet being used with ImageNet)
Notes from Deep Residual Learning for Image Recognition by He et al:

  The paper used ImageNet images, but we are using CIFAR100.  However, we will
  still follow their practice of random cropping and horizontal flipping.
  Per pixel mean is subtracted as well.
  We will not perform color augmentation to stay consistent with our data
  augmentation done with VGG16.

  Batch norm is performed after each convolution and before the activation
  function.

  Weight are initialized using he_uniform.

  Optimization is done using SGD with a mini-batch size of 256.  The learning
  rate starts from 0.1 and is divided by 10 when the error plateaus.

  They use a weight decay of 0.0001 and momentum of 0.9.

  Their model is trained for 60x10^4 iterations.


(The above paper includes a separate architecture for use with Cifar)
ResNet with Cifar:

  The initial layer is a 3x3 conv2d with 16 filters, stride 1, padding 1

  The basic blocks are the same within the architecture, but how often they are
  used changes.  So functions basic_block and make_layer did not need to be
  modified (bottleneck is not used at all for Cifar).  Most changes are made in
  def resnet, where the initial layer is modified and how to build the blocks.

  For data augmentation, 4 pixels are padded on each side and then the random
  crop and horizontal flip are performed.
  
  Batch size of 128 is used.

"""
# If the residual connection is not the same shape as F(x), modify the shape
  # of the residual


# Previously (as seen in VGG16), we used model.add() to build nets, which is the
# Sequential API in Keras.  This code uses the Functional API, which uses x to
# represent the current state, and passes it as input to the new layer being
# called.


# Basic block for 18 and 34 layer ResNets
# Basic block is the same for Cifar application (6n+2 layers)
def basic_block(x, filters, act_funcs, kernel_init, stride=1):

    #print("basic block")
    # Save the input for the skip connection
    residual = x

    # print("x shape: ", x.shape)
    # 3x3 Convolution
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                      use_bias=False, kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = CheckNumericsLayer("in basic block, before blend 1")(x)
    x = adaptive_blending(act_funcs)(x)
    x = CheckNumericsLayer("in basic block, after blend 1")(x)

    # 3x3 Convolution
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    # print("x shape: ", x.shape)
    # print("residual shape: ", residual.shape)
    # check that the dimensions of F(x) and x (the residual) match
    if x.shape[-1] != residual.shape[-1]:
      # print("up or down sampling")
      # if they don't match, modify the residual so that they do match
      residual = layers.Conv2D(x.shape[-1], kernel_size=1, strides=stride,
                               padding='same', use_bias=False,
                               kernel_initializer=kernel_init)(residual)
      residual = layers.BatchNormalization()(residual)
      # print("residual shape: ", residual.shape)


    # Add the shortcut connection
    x = layers.add([x, residual])
    x = CheckNumericsLayer(f"in basic block, before blend 2")(x)
    x = adaptive_blending(act_funcs)(x)
    x = CheckNumericsLayer(f"in basic block, after blend 2")(x)

    return x


# Bottleneck block for 50, 101, and 152 layer ResNets
def bottleneck(x, filters, act_funcs, kernel_init, stride=1):
    # print('bottleneck')

    # Save the input for the skip connection
    residual = x

    # print('F(X) shape: ', x.shape)
    # 1x1 Convolution (reduce dimensions)
    x = layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False,
                      kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = adaptive_blending(act_funcs)(x)

    # print('F(X) shape after 1x1 conv: ', x.shape)
    # 3x3 Convolution (bottleneck layer)
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = adaptive_blending(act_funcs)(x)

    # print('F(X) shape after 3x3 conv: ', x.shape)
    # 1x1 Convolution (restore dimensions)
    x = layers.Conv2D(filters*4, kernel_size=1, strides=1, use_bias=False,
                      kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)

    # print('F(X) shape after 1x1 conv: ', x.shape)

    # Check that F(x) and x are of the same shape
    # If they are not the same shape, modify the dimensions of the residual
    # to match
    # print('F(X) shape: ', x.shape[-1])
    # print('residual shape: ', residual.shape[-1])
    if x.shape[-1] != residual.shape[-1]:

      # print('upsampling...')
      # use a 1x1 convolution to match the dimensions
      residual = layers.Conv2D(x.shape[-1], kernel_size=1, strides=stride,
                               padding='same', use_bias=False,
                               kernel_initializer=kernel_init)(residual)
      residual = layers.BatchNormalization()(residual)

      #print(residual.shape)

    # Add the residual connection
    x = layers.add([x, residual])
    x = adaptive_blending(act_funcs)(x)

    return x


# function to specify how to build blocks
def _make_layer(x, filters, blocks, act_funcs, block_type, kernel_init, stride=1):

    # Build each set of layers within a block
      # build that block the number of times specified by blocks
    # the last three sets of blocks have stride=2 for the first conv2d, so the
      # first block is built separately to accomodate that

    # basic is for ResNet18 and 34
    # basic is also used for cifar application
    if block_type == 'basic':
      # make the first block
      x = basic_block(x, filters, act_funcs, kernel_init, stride=stride)
      # make the rest of the blocks
      for _ in range(1, blocks):
        x = basic_block(x, filters, act_funcs, kernel_init)

    # otherwise it is a larger ResNet (50, 101, or 152)
    else:
      # build the first layer
      x = bottleneck(x, filters, act_funcs, kernel_init, stride=stride)
      # build the rest of the blocks
      for _ in range(1, blocks):
        x = bottleneck(x, filters, act_funcs, kernel_init)

    return x


# Base function to build ResNet for Imagenet
def resnet(input_shape, num_classes, num_blocks, act_funcs, block_type, kernel_init):

    # when using ImageNet, num_blocks replaces depth

    # set the inputs
    inputs = layers.Input(shape=input_shape)

    # Initial convolution and maxpooling layers
    # 16 filters, 3x3 kernel, stride 1 (for Cifar)
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same',
                      use_bias=False, 
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = CheckNumericsLayer("before activation, first layer")(x)
    x = adaptive_blending(act_funcs)(x)
    x = CheckNumericsLayer("After activation, first layer")(x)
    # NO MAX POOL WHEN USING CIFAR
    #x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual Blocks (Stage 1 to Stage 4) FOR IMAGENET
    # all resnet sizes use 4 blocks, with same base filters
    # In the last 3 blocks, downsampling is done with a stride of 2
      # stride of 2 is only used on the first convolution
    x = _make_layer(x, 64, num_blocks[0], act_funcs, block_type, kernel_init)
    x = _make_layer(x, 128, num_blocks[1], act_funcs, block_type, kernel_init,
                    stride=2)
    x = _make_layer(x, 256, num_blocks[2], act_funcs, block_type, kernel_init,
                    stride=2)
    x = _make_layer(x, 512, num_blocks[3], act_funcs, block_type, kernel_init,
                    stride=2)

    # Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer (Dense Layer)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs, x)

    return model



# Base function to build ResNet for Cifar
def resnetcifar(input_shape, num_classes, depth, act_funcs, block_type, kernel_init):

    # when using Cifar, depth replaces num_blocks
    assert (depth - 2) % 6 == 0, "Depth should be 6n+2 (e.g., 20, 32, 44, 56, 110)"
    n = (depth - 2) // 6


    # set the inputs
    inputs = layers.Input(shape=input_shape)

    # Initial convolution and maxpooling layers
    # 16 filters, 3x3 kernel, stride 1 (for Cifar)
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same',
                      use_bias=False, 
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = adaptive_blending(act_funcs)(x)

    # Residual Blocks (Stage 1 to Stage 4) FOR CIFAR
    # all resnet sizes use 3 blocks, with same base filters
    x = _make_layer(x, 16, n, act_funcs, block_type, kernel_init)
    x = _make_layer(x, 32, n, act_funcs, block_type, kernel_init, stride=2)
    x = _make_layer(x, 64, n, act_funcs, block_type, kernel_init, stride=2)


    # Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer (Dense Layer)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs, x)

    return model


### set up learning rate scheduler
"""
The learning rate starts at 0.1 and is divided by 10 at 32k and 48k iterations.
epochs = iterations / (training set size / batch size)
So for 32k and 48k iterations that is epoch 91 and 137
"""

def lr_scheduler(epoch):
  # initial learning rate defined at top
  if epoch < epoch1:
    return lr_start
  elif epoch < epoch2:
    return lr_start / 10
  else:
    return lr_start / 100




def set_all_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    return





#%%
### Load dataset and set up data augmentation


# load CIFAR100 data
(x_train_cf100, y_train_cf100), (x_test_cf100, y_test_cf100) = tf.keras.datasets.cifar100.load_data()
x_train, y_train, x_test, y_test = x_train_cf100, y_train_cf100, x_test_cf100, y_test_cf100

# normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# subtract the mean of each channel
# calculated elsewhere
x_train = x_train - ([0.5071, 0.4865, 0.4409])
x_test = x_test - ([0.5071, 0.4865, 0.4409])

# split up the training data into some validation data
val_split = 0.1
val_loc = int(len(x_train)*val_split)

x_val, y_val = x_train[:val_loc], y_train[:val_loc]
x_train, y_train = x_train[val_loc:], y_train[val_loc:]

# convert y labels to one hot encoding
y_train = to_categorical(y_train, num_classes=100)
y_val = to_categorical(y_val, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)



# Data augmentation
# layers.ZeroPadding2D doesn't always work with the data augmentation pipeline,
  # so we will set up the padding as a lambda instead
data_augmentation = tf.keras.Sequential([
  layers.Lambda(lambda x: tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT')),
  layers.RandomFlip("horizontal"),
  layers.RandomCrop(32, 32)
])


# only convert data format and batch it
# training data will be randomly shuffled & augmented for each seed
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(64)



#%%
### Train model(s) and save and plot the results


# store model.evaulate results 	
eval_results = []

# train model num_runs times with different seeds
for i in range(num_runs):
    
    # generate a random seed value (0 to 32 bits) and set seeds
    seed_value = random.randint(0, 2**32)  
    set_all_seeds(seed_value)
    
    # make folder to save results in
    # exit the program if the folder already exists
    folder = f'run_{i}'
    if os.path.exists(folder):
        print(f"\nError: {folder} directory exists.  Move or delete files.  Exiting program.")
        sys.exit(1)
    else:
        os.makedirs(folder)
    
    ### Perform data augmentation

    # Convert images to tf.data.Dataset format
    # from_tensor_slices creates the Dataset from data whose elements are the slices
      # of the tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # shuffle the data, batch it, and then augment the images
    train_ds = train_ds.shuffle(45000)
    train_ds = train_ds.batch(64)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))


    
    ### set up and train models


    
    # Define number of blocks for each model
    print('running')
    
    
    ### build the model
    # Built for CIFAR100
    # use num_blocks with ImageNet, depth with Cifar
      # num_blocks = res18
    #model = resnetcifar(input_shape=(32, 32, 3), num_classes=100, depth=depth,
    #               block_type=block_type, act_func=act_func, kernel_init=kernel_init)
    model = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                   block_type=block_type, act_funcs=act_funcs, kernel_init=kernel_init)
    
    # Print the model summary
    # model.summary()

    
    # Compile and fit the model
    # since we manually added weight decay in the model def, we do not declare
        # it here (otherwise it will double penalize the loss)
    opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9, clipnorm=5.0)
    #opt = optimizers.Adadelta(learning_rate=lr_start)
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # set up learning rate scheduler
    lrate = LearningRateScheduler(lr_scheduler)
    alpha_logs = alpha_logger(output_dir=os.path.join(folder, "alpha_logs"))

    
    start_time = time.time()
    history = model.fit(train_ds,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(val_ds), callbacks=[lrate, alpha_logs])
    
    
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
    
    ### For use with individual model
    # plotting an individual activation function's loss and accuracy
    
    # Evaluate the model with the test dataset
    test_results = model.evaluate(test_ds, return_dict=True)
    eval_results.append({
        'run': i,
        'seed': seed_value, 
        'accuracy': test_results['accuracy'],
        'loss': test_results['loss']
        })
    # np.save(os.path.join(folder, f'{act_func}_test_acc_loss.npy'), test_results)
    # print(f"Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}")
    
    ### Plots
    ### Plot training vs. validation for individual act. funcs.
    
    # ReLU
    plt.figure(1, figsize=(12, 5))
    
    # Accuracy: training vs. validation plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'ABU ResNet{net_depth} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss: training vs. validation plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'ABU ResNet{net_depth} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(folder, f'ABU ResNet{net_depth} plot.png'))
    plt.close()
    
    
    # save log of history using numpy
    np.save(os.path.join(folder, f"abu_history_{i}.npy"), history.history)
    # save test eval results per run
    np.save(os.path.join(folder, f"eval_results_{i}.npy"), eval_results) 
    # to load:
    # history = np.load('file.npy', allow_pickle='TRUE').item()
    # note that when referencing history after loading, it is no longer necessary to use 
    # history.history['item'], but only history['item'] is needed
        
        
        
# save the model evaluation(s) and print results
folder = 'eval_results'
os.makedirs(folder)
np.save(os.path.join(folder, 'eval_results.npy'), eval_results) 

# Note for loading that these results are a list of dictionaries (when doing multiple runs)
# so you need to use np.load('file.npy', allow_pickle='TRUE').tolist()


# compute and print averages and variances of test results per function
print(eval_results)

# make np.array of loss and accuracy
accuracies = np.array([r['accuracy'] for r in eval_results])
losses = np.array([r['loss'] for r in eval_results])

# compute and print the average, variance and standard deviation of the accuracy and loss
acc_mean = np.mean(accuracies)
acc_var = np.var(accuracies, ddof=1)
acc_std = np.std(accuracies, ddof=1)
loss_mean = np.mean(losses)
loss_var = np.var(losses, ddof=1)
loss_std = np.std(losses, ddof=1)

print(f"Results from {num_runs} runs: ")
print(f"  Accuracy: mean = {acc_mean:.4f}, variance = {acc_var:.6f}, standard dev. = {acc_std: .6f}")
print(f"  Loss:     mean = {loss_mean:.4f}, variance = {loss_var:.6f}, standard dev. = {loss_std: .6f}")
print()

