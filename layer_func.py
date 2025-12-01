
"""
This layer reads in the .csv files for the adaptive_blending layers.
It is hard coded to read in filenames, so at line 101 select which set of
files you need to read from.
It plots the resulting function of each blended layer and saves it in the
current working directory.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf



# list of filenames for run 0
filenames0 = [
    "adaptive_blending_alphas",
    "adaptive_blending_1_alphas",
    "adaptive_blending_2_alphas",
    "adaptive_blending_3_alphas",
    "adaptive_blending_4_alphas",
    "adaptive_blending_5_alphas",
    "adaptive_blending_6_alphas",
    "adaptive_blending_7_alphas",
    "adaptive_blending_8_alphas",
    "adaptive_blending_9_alphas",
    "adaptive_blending_10_alphas",
    "adaptive_blending_11_alphas",
    "adaptive_blending_12_alphas",
    "adaptive_blending_13_alphas",
    "adaptive_blending_14_alphas",
    "adaptive_blending_15_alphas",
    "adaptive_blending_16_alphas"
]

filenames1 = [
    "adaptive_blending_17_alphas",
    "adaptive_blending_18_alphas",
    "adaptive_blending_19_alphas",
    "adaptive_blending_20_alphas",
    "adaptive_blending_21_alphas",
    "adaptive_blending_22_alphas",
    "adaptive_blending_23_alphas",
    "adaptive_blending_24_alphas",
    "adaptive_blending_25_alphas",
    "adaptive_blending_26_alphas",
    "adaptive_blending_27_alphas",
    "adaptive_blending_28_alphas",
    "adaptive_blending_29_alphas",
    "adaptive_blending_30_alphas",
    "adaptive_blending_31_alphas",
    "adaptive_blending_32_alphas",
    "adaptive_blending_33_alphas"
]

filenames2 = [
    "adaptive_blending_34_alphas",
    "adaptive_blending_35_alphas",
    "adaptive_blending_36_alphas",
    "adaptive_blending_37_alphas",
    "adaptive_blending_38_alphas",
    "adaptive_blending_39_alphas",
    "adaptive_blending_40_alphas",
    "adaptive_blending_41_alphas",
    "adaptive_blending_42_alphas",
    "adaptive_blending_43_alphas",
    "adaptive_blending_44_alphas",
    "adaptive_blending_45_alphas",
    "adaptive_blending_46_alphas",
    "adaptive_blending_47_alphas",
    "adaptive_blending_48_alphas",
    "adaptive_blending_49_alphas",
    "adaptive_blending_50_alphas"
]

# list of activation functions
# in order as they appear in .csv files
act_funcs = [
    tf.keras.activations.relu,
    tf.keras.activations.leaky_relu,
    tf.keras.activations.sigmoid,
    tf.keras.activations.tanh,
    tf.keras.activations.silu,
    tf.keras.activations.elu,
    tf.keras.activations.selu,
    tf.keras.activations.gelu
]



# define x range for plotting function
x = np.linspace(-10, 10, 1000)
# convert to tensor to use with activation functions
x = tf.convert_to_tensor(x)

# load all csv files and plot the final alpha
n = 0
for file_i in filenames0:

    # load file and get layer number and alpha value
    filename = f"{file_i}.csv"
    df = pd.read_csv(filename)
    # the first column is the epoch, so cut that off
    final_row = df.iloc[-1, 1:]
    functions = range(len(final_row))
    alphas = final_row.values.astype(float)
    
    # generate resulting function 
    ABU_out = np.zeros_like(x)
    for j in range(len(alphas)):
        ABU_out += alphas[j] * act_funcs[j](x)

    
    # make plot of final layer function
    plt.figure(figsize=(6,4))
    plt.plot(x, ABU_out.numpy())
    plt.title(f"Final function of layer {n}")
    plt.xlabel('x')
    plt.ylabel('layer output')
    plt.tight_layout()
    plt.savefig(f"layer_{n}_function.png")
    plt.close()
    
    # update layer number
    n+=1





