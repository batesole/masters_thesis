# -*- coding: utf-8 -*-
"""
ResNet with CIFAR-100

ResNet architecture to be used with CIFAR-100.  Program loads the dataset using
tensorflow, so it does not need to be downloaded separately.

This program includes architecture to build ResNet based on both the ImageNet
style architecture and Cifar style architecture.  However, it should be noted that
the final architecture used for experiments used the ImageNet style with the first 
global max pool removed, so that layer is commented out.

The README in the github repository includes an explanation of the global parameters.

This program trains the architecture using a random seed each run.  Each run creates 
its own folder to save history plots and numpy files of the model evaluations.  At 
the end of all the runs, the average loss and accuracy is printed out for each 
activation function.


"""

import time
import datetime
import numpy as np
import random
import os
from collections import defaultdict
import tensorflow as tf
import keras

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
# define how many models we want to run and compare
num_models = 8

### Model related parameters
# learning rate, which epochs to change the learning rate, and weight decay
# lr scheduler defined after model def
lr_start = 0.1
epoch1 = 40
epoch2 = 80
weight_decay = 0.0001
# Define epochs and batch size
epochs = 120
batch_size = 128

# ResNet sizes for use with ImageNet
# UNCOMMENT block_type FOR THE RESPECTIVE SIZE
# resnets 18 and 34 have smaller block types
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
num_blocks = res34
# network depth is used for making the titles of plots
net_depth = 34

# ResNet size for use with Cifar (6n+2 = 20, 32, 44, 56, 110, 1202)
depth = 56

# define the activation function to use (for single model run)
# relu, tanh, sigmoid, leaky_relu, elu, selu, gelu
# he_uniform for relu family, glorot_uniform for sigmoid family
# lecun_uniform for elu family
act_func = 'relu'
kernel_init = 'he_uniform'


# define the number of times to the model 
# each run will use a random seed
num_runs = 5







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
def basic_block(x, filters, act_func, kernel_init, stride=1):

    # print("basic block")
    # Save the input for the skip connection
    residual = x

    # print("x shape: ", x.shape)
    # 3x3 Convolution
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same',
                      use_bias=False, kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_func)(x)

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
    x = layers.Activation(act_func)(x)

    return x


# Bottleneck block for 50, 101, and 152 layer ResNets
def bottleneck(x, filters, act_func, kernel_init, stride=1):
    # print('bottleneck')

    # Save the input for the skip connection
    residual = x

    # print('F(X) shape: ', x.shape)
    # 1x1 Convolution (reduce dimensions)
    x = layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False,
                      kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_func)(x)

    # print('F(X) shape after 1x1 conv: ', x.shape)
    # 3x3 Convolution (bottleneck layer)
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      use_bias=False, kernel_initializer=kernel_init,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_func)(x)

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
    x = layers.Activation(act_func)(x)

    return x


# function to specify how to build blocks
def _make_layer(x, filters, blocks, act_func, block_type, kernel_init, stride=1):

    # Build each set of layers within a block
      # build that block the number of times specified by blocks
    # the last three sets of blocks have stride=2 for the first conv2d, so the
      # first block is built separately to accomodate that

    # basic is for ResNet18 and 34
    # basic is also used for cifar application
    if block_type == 'basic':
      # make the first block
      x = basic_block(x, filters, act_func, kernel_init, stride=stride)
      # make the rest of the blocks
      for _ in range(1, blocks):
        x = basic_block(x, filters, act_func, kernel_init)

    # otherwise it is a larger ResNet (50, 101, or 152)
    else:
      # build the first layer
      x = bottleneck(x, filters, act_func, kernel_init, stride=stride)
      # build the rest of the blocks
      for _ in range(1, blocks):
        x = bottleneck(x, filters, act_func, kernel_init)

    return x


# Base function to build ResNet for Imagenet
def resnet(input_shape, num_classes, num_blocks, act_func, block_type, kernel_init):

    # when using ImageNet, num_blocks replaces depth

    # set the inputs
    inputs = layers.Input(shape=input_shape)

    # Initial convolution and maxpooling layers
    # 16 filters, 3x3 kernel, stride 1 (for Cifar)
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same',
                      use_bias=False, 
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act_func)(x)
    # NO MAX POOL WHEN USING CIFAR
    #x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual Blocks (Stage 1 to Stage 4) FOR IMAGENET
    # all resnet sizes use 4 blocks, with same base filters
    # In the last 3 blocks, downsampling is done with a stride of 2
      # stride of 2 is only used on the first convolution
    x = _make_layer(x, 64, num_blocks[0], act_func, block_type, kernel_init)
    x = _make_layer(x, 128, num_blocks[1], act_func, block_type, kernel_init,
                    stride=2)
    x = _make_layer(x, 256, num_blocks[2], act_func, block_type, kernel_init,
                    stride=2)
    x = _make_layer(x, 512, num_blocks[3], act_func, block_type, kernel_init,
                    stride=2)

    # Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer (Dense Layer)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs, x)

    return model



# Base function to build ResNet for Cifar
def resnetcifar(input_shape, num_classes, depth, act_func, block_type, kernel_init):

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
    x = layers.Activation(act_func)(x)

    # Residual Blocks (Stage 1 to Stage 4) FOR CIFAR
    # all resnet sizes use 3 blocks, with same base filters
    x = _make_layer(x, 16, n, act_func, block_type, kernel_init)
    x = _make_layer(x, 32, n, act_func, block_type, kernel_init, stride=2)
    x = _make_layer(x, 64, n, act_func, block_type, kernel_init, stride=2)


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
    folder = f'run_{i}'
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
    
    # Run multiple models each with a different activation function
    if num_models >1:
        
        # Define number of blocks for each model
        print('running')
        
        
        ### build the models
        
        # Create a separate model for each activation function
        model_relu = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='relu', kernel_init='he_uniform')
        model_lrelu = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='leaky_relu', kernel_init='he_uniform')
        model_sig = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='sigmoid', kernel_init='glorot_uniform')
        model_tanh = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='tanh', kernel_init='glorot_uniform')
        model_silu = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='silu', kernel_init='glorot_uniform')
        model_elu = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='elu', kernel_init='lecun_uniform')
        model_selu = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='selu', kernel_init='lecun_uniform')
        model_gelu = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func='gelu', kernel_init='lecun_uniform')
        
        
        # Print the model summary
        # model.summary()
        
        ### Run the models
        
        # Set up optimizer, compile and train each model
        # since we manually added weight decay in the model def, we do not declare
            # it here (otherwise it will double penalize the loss)
        start_time = time.time()
        # set up learning rate scheduler
        lrate = LearningRateScheduler(lr_scheduler)
                            
        
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_relu.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_relu = model_relu.fit(train_ds,
                                      epochs=epochs, batch_size=batch_size,
                                      validation_data=(val_ds), callbacks=[lrate])
        relu_time = time.time() - start_time
        
        lrelu_time = time.time()
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_lrelu.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_lrelu = model_lrelu.fit(train_ds,
                                        epochs=epochs, batch_size=batch_size,
                                        validation_data=(val_ds), callbacks=[lrate])
        lrelu_time = time.time() - lrelu_time
        
        sig_time = time.time()
        # sigmoid needs more weight decay to converge properly, so we will add another
        # weight decay here to double penalize it
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_sig.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_sig = model_sig.fit(train_ds,
                                    epochs=epochs, batch_size=batch_size,
                                    validation_data=(val_ds), callbacks=[lrate])
        sig_time = time.time() - sig_time
        
        tanh_time = time.time()
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_tanh.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_tanh = model_tanh.fit(train_ds,
                                      epochs=epochs, batch_size=batch_size,
                                      validation_data=(val_ds), callbacks=[lrate])
        tanh_time = time.time() - tanh_time
        
        silu_time = time.time()
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_silu.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_silu = model_silu.fit(train_ds,
                                      epochs=epochs, batch_size=batch_size,
                                      validation_data=(val_ds), callbacks=[lrate])
        silu_time = time.time() - silu_time
        
        elu_time = time.time()
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_elu.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_elu = model_elu.fit(train_ds,
                                    epochs=epochs, batch_size=batch_size,
                                    validation_data=(val_ds), callbacks=[lrate])
        elu_time = time.time() - elu_time
        
        selu_time = time.time()
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_selu.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_selu = model_selu.fit(train_ds,
                                      epochs=epochs, batch_size=batch_size,
                                      validation_data=(val_ds), callbacks=[lrate])
        selu_time = time.time() - selu_time
        
        gelu_time = time.time()
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        model_gelu.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history_gelu = model_gelu.fit(train_ds,
                                      epochs=epochs, batch_size=batch_size,
                                      validation_data=(val_ds), callbacks=[lrate])
        gelu_time = time.time() - gelu_time
        
        
        # print the run time
        end_time = time.time()
        run_time = end_time - start_time
        print(run_time)
        
        
        
        ### Print the test accuracy and loss of each model
        
        # Evaluate each model with the test dataset and save the results
        # test_results = model_relu.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'relu_test_acc_loss.npy'), test_results)
        # print(f"ReLU Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {relu_time} seconds")
        
        # test_results = model_lrelu.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'leaky_relu_test_acc_loss.npy'), test_results)
        # print(f"Leaky ReLU Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {lrelu_time} seconds")
        
        # test_results = model_sig.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'sigmoid_test_acc_loss.npy'), test_results)
        # print(f"Sigmoid Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {sig_time} seconds")
        
        # test_results = model_tanh.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'tanh_test_acc_loss.npy'), test_results)
        # print(f"Tanh Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {tanh_time} seconds")
        
        # test_results = model_silu.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'silu_test_acc_loss.npy'), test_results)
        # print(f"SILU Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {silu_time} seconds")
        
        # test_results = model_elu.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'elu_test_acc_loss.npy'), test_results)
        # print(f"ELU Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {elu_time} seconds")
        
        # test_results = model_selu.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'selu_test_acc_loss.npy'), test_results)
        # print(f"SELU Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {selu_time} seconds")
        
        # test_results = model_gelu.evaluate(test_ds, return_dict=True)
        # np.save(os.path.join(folder, 'gelu_test_acc_loss.npy'), test_results)
        # print(f"GELU Test Loss: {test_results['loss']}, Test Accuracy: {test_results['accuracy']}, Run Time: {gelu_time} seconds")
        
        
        
        ### Plots
        
        ### Individual Plots
        # Plot training vs. validation for individual act. funcs.
        act_func_list = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'silu', 'elu', 'selu', 'gelu']
        history_list = [history_relu, history_lrelu, history_sig, history_tanh, history_silu, history_elu, history_selu, history_gelu]
        model_list = [model_relu, model_lrelu, model_sig, model_tanh, model_silu, model_elu, model_selu, model_gelu]
        
        for k in range(len(history_list)):
        
            # set up figure
            plt.figure(figsize=(12, 5))
    	    
            # Accuracy: training vs. validation plot
            plt.subplot(1, 2, 1)
            plt.plot(history_list[k].history['accuracy'], label='Train Accuracy')
            plt.plot(history_list[k].history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{act_func_list[k]} ResNet{net_depth} Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
    	    
    	# Loss: training vs. validation plot
            plt.subplot(1, 2, 2)
            plt.plot(history_list[k].history['loss'], label='Train Loss')
            plt.plot(history_list[k].history['val_loss'], label='Validation Loss')
            plt.title(f'{act_func_list[k]} ResNet{net_depth} Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
    	  
            plt.tight_layout()
    	#plt.show()
            plt.savefig(os.path.join(folder, f'{act_func_list[k]} ResNet{net_depth} plot.png'))
            plt.close()
            
            # save log of history using numpy
            np.save(os.path.join(folder,f'{act_func_list[k]}_history.npy'), history_list[k].history)
            # to load:
            # history = np.load('file.npy', allow_pickle='TRUE').item()
            # note that when referencing history after loading, it is no longer necessary to use 
            # history.history['item'], but only history['item'] is needed
            
            # evaluate the model and save the results
            test_results = model_list[k].evaluate(test_ds, return_dict=True)
            eval_results.append({
                'run': i,
                'seed': seed_value,
                'function': act_func_list[k],
                'accuracy': test_results['accuracy'],
                'loss': test_results['loss']
                })
        
        
        ### All Functions Plots
        # First plot is training accuracy, second plot is training loss
        plt.figure(1)
        
        # Accuracy: training vs. validation plot
        # plt.subplot(1, 2, 1)
        plt.plot(history_relu.history['val_accuracy'], label='ReLU Accuracy')
        plt.plot(history_lrelu.history['val_accuracy'], label='Leaky ReLU Accuracy')
        plt.plot(history_sig.history['val_accuracy'], label='Sigmoid Accuracy')
        plt.plot(history_tanh.history['val_accuracy'], label='Tanh Accuracy')
        plt.plot(history_silu.history['val_accuracy'], label='SILU Accuracy')
        plt.plot(history_elu.history['val_accuracy'], label='ELU Accuracy')
        plt.plot(history_selu.history['val_accuracy'], label='SELU Accuracy')
        plt.plot(history_gelu.history['val_accuracy'], label='GELU Accuracy')
        plt.title(f'Validation Accuracies with ResNet{net_depth} Model')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        #plt.show()
        plt.savefig(os.path.join(folder, 'multi_acc_plots.png'))
        plt.close()
        
        plt.figure(2)
        # Loss: training vs. validation plot
        # plt.subplot(1, 2, 2)
        plt.plot(history_relu.history['val_loss'], label='ReLU Loss')
        plt.plot(history_lrelu.history['val_loss'], label='Leaky ReLU Loss')
        plt.plot(history_sig.history['val_loss'], label='Sigmoid Loss')
        plt.plot(history_tanh.history['val_loss'], label='Tanh Loss')
        plt.plot(history_silu.history['val_loss'], label='SILU Loss')
        plt.plot(history_elu.history['val_loss'], label='ELU Loss')
        plt.plot(history_selu.history['val_loss'], label='SELU Loss')
        plt.plot(history_gelu.history['val_loss'], label='GELU Loss')
        plt.title(f'Validation losses with ResNet{net_depth} Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(folder,'multi_loss_plots.png'))
        plt.close()
        
        # save test eval results per run
        np.save(os.path.join(folder, f"eval_results_{i}.npy"), eval_results) 
    
    
    
    #%%
    ### Running only one model
    
    if num_models == 1:
        
        # Define number of blocks for each model
        print('running')
        
        
        ### build the model
        # Built for CIFAR100
        # use num_blocks with ImageNet, depth with Cifar
          # num_blocks = res18
        #model = resnetcifar(input_shape=(32, 32, 3), num_classes=100, depth=depth,
        #               block_type=block_type, act_func=act_func, kernel_init=kernel_init)
        model = resnet(input_shape=(32, 32, 3), num_classes=100, num_blocks=num_blocks,
                       block_type=block_type, act_func=act_func, kernel_init=kernel_init)
        
        # Print the model summary
        # model.summary()
    
        
        # Compile and fit the model
        # since we manually added weight decay in the model def, we do not declare
            # it here (otherwise it will double penalize the loss)
        opt = optimizers.SGD(learning_rate=lr_start, momentum=0.9)
        #opt = optimizers.Adadelta(learning_rate=lr_start)
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # set up learning rate scheduler
        lrate = LearningRateScheduler(lr_scheduler)
    
        
        start_time = time.time()
        history = model.fit(train_ds,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(val_ds), callbacks=[lrate])
        
        
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
            'function': act_func,
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
        plt.title(f'{act_func} ResNet{net_depth} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss: training vs. validation plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{act_func} ResNet{net_depth} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(folder, f'{act_func} ResNet{net_depth} plot.png'))
        plt.close()
        
        
        # save log of history using numpy
        np.save(os.path.join(folder, 'single_history.npy'), history.history)
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

# group results by activation function
groups = defaultdict(list)
for result in eval_results:
    func = result['function']
    groups[func].append(result)
    
# compute and print averages and variances of test results per function
for func,results in groups.items():
    
    # for each activation function, make np.array of loss and accuracy
    accuracies = np.array([r['accuracy'] for r in results])
    losses = np.array([r['loss'] for r in results])
    
    # compute and print the average, variance and standard deviation of the accuracy and loss
    acc_mean = np.mean(accuracies)
    acc_var = np.var(accuracies, ddof=1)
    acc_std = np.std(accuracies, ddof=1)
    loss_mean = np.mean(losses)
    loss_var = np.var(losses, ddof=1)
    loss_std = np.std(losses, ddof=1)
    
    print(f"Function: {func}")
    print(f"  Accuracy: mean = {acc_mean:.4f}, variance = {acc_var:.6f}, standard dev. = {acc_std: .6f}")
    print(f"  Loss:     mean = {loss_mean:.4f}, variance = {loss_var:.6f}, standard dev. = {loss_std: .6f}")
    print()

