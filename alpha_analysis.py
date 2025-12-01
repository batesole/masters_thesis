"""
This file will perform an analysis of the resulting alpha weights from
the function blending.

Since the program that trains the network already creates plots of the
weights per layer and per function, we will not do that here.

Steps: 
    1 - Load in all .csv files in a defined parent directory.  The parent directory 
    is hard coded in this file.
    
    - PARENT DIRECTORY - 
    |- Experiment 1
    |  |- Run 1
    |  |  |- alphas
    |  |  |  |- layer1_alphas.csv
    |  |  |  |- func1_alphas.csv
    |  |  |- ...
    |  |- ...
    |  |- Run n
    |  |  |- alpha csv files
    |- Experiment ...
    |  |- run ...
    |  |  |- alpha
    |  |  |  |- alpha csv... 
    
    
    
    

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict
from scipy import stats


parent_dir = "data"


#%% FUNCTION DEFINITIONS

# function to load all .csv files from given parent directory
def load_all_weights(parent_dir):
    # load all data into separate dictionaries
    data_func = defaultdict(list)
    data_layer = defaultdict(list)
    
    final_alpha_func = defaultdict(list)
    final_alpha_layer = defaultdict(list)
    
    # there are multiple experiment types
    for experiment_i in os.listdir(parent_dir):
        experiment_path = os.path.join(parent_dir, experiment_i)        
        # verify the experiment path is actually a folder
        if not os.path.isdir(experiment_path):
            # skip this file if it's not a folder
            continue
        
        # go through the runs in the current experiment
        for run_i in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run_i)
            if not os.path.isdir(run_path):
                continue
            
            # go through all csv files in the current run folder
            # the csv files are located inside a folder called alpha_logs
            run_path = os.path.join(run_path, "alpha_logs")
            if not os.path.isdir(run_path):
                continue
            for file_i in os.listdir(run_path):
                if not file_i.endswith(".csv"):
                    continue
                filepath = os.path.join(run_path, file_i)
                
                # the csv files follow naming format adaptive_blending_x_alphas.csv
                # or functionname_alphas.csv
                # so we can get the first word to determine if the file is a layer or function
                # splitext splits the filename text into the name and extension
                base_name = os.path.splitext(file_i)[0]
                
                # rows are epochs, columns are alphas per layer or function
                df = pd.read_csv(filepath)
                
                # we will use de.melt to make plotting easier
                df_melt = df.melt(ignore_index = False )
                
                # check if we are looking at a layer 
                if base_name.startswith("adaptive"):
                    # identifier = base_name.replace("adaptive_blending", "ABU_layer")
                    # identifier = identifier.replace("_alphas", "")
                    identifier = "ABU_layer"
                    
                    # extract the layer number
                    # layer 0 does not include the 0, so check if there is a digit first
                    	# filter(function, iterable)
                    digit_num = ''.join(filter(str.isdigit, base_name))
                    if digit_num:
                        layer_num = int(digit_num)
                    else:
                        layer_num = 0
                    
                    # now extract the run number 
                    # run0 has layers 0-16, run1 17-33, etc
                    # normalize them to all show as layers 0-16
                    run_num = int(''.join(filter(str.isdigit, run_i)))
                    layer_norm = layer_num - (run_num * 17)
                    
                    
                    layer = f"{identifier}_{layer_norm}"
                    df_melt["experiment"] = experiment_i
                    df_melt["run_id"] = run_i
                    df_melt["layer"] = layer
                    df_melt["file_path"] = filepath
                    
                    # append to all data by layer
                    data_layer[f"{experiment_i}_{run_i}_{layer}"].append(df_melt)
                    
                    
                    # now pull in final alphas for a separate df
                    final_row = df.iloc[-1, 1:]
                    epoch_val = df.iloc[-1, 0]
                    
                    df_final_melt = final_row.reset_index()
                    df_final_melt.columns = ["function", "alpha"]
                    df_final_melt["epoch"] = epoch_val
                    df_final_melt["experiment"] = experiment_i
                    df_final_melt["run_id"] = run_i
                    # df_final_melt["function"] = function
                    df_final_melt["file_path"] = filepath
                    
                    # append to final data by layer
                    final_alpha_layer[layer].append(df_final_melt)
                    
                    
                    
                    
                # if it's not a layer then it's a function
                else:
                    identifier = base_name.replace("_alphas", "")
                    
                    # extract the layer number
                    # layer 0 does not include the 0, so check if there is a digit first
                    	# filter(function, iterable)
                    # layer_str = df["layer"] 
                    # digit_num = ''.join(filter(str.isdigit, layer_str))
                    # if digit_num:
                    #     layer_num = int(digit_num)
                    # else:
                    #     layer_num = 0
                    
                    # now extract the run number 
                    # run0 has layers 0-16, run1 17-33, etc
                    # normalize them to all show as layers 0-16
                    # run_num = int(''.join(filter(str.isdigit, run_i)))
                    # layer_norm = layer_num - (run_num * 17)
                    
                    function = identifier
                    df_melt["experiment"] = experiment_i
                    df_melt["run_id"] = run_i
                    df_melt["function"] = function
                    df_melt["file_path"] = filepath
                    # df_melt["layer"] = layer_norm
                    
                    data_func[function].append(df_melt)
                    
                    # pull in the final alpha values
                    final_row = df.iloc[-1, 1:]
                    epoch_val = df.iloc[-1, 0]
                    
                    df_final_melt = final_row.reset_index()
                    df_final_melt.columns = ["layer", "alpha"]
                    df_final_melt["epoch"] = epoch_val
                    df_final_melt["experiment"] = experiment_i
                    df_final_melt["run_id"] = run_i
                    df_final_melt["function"] = function
                    df_final_melt["file_path"] = filepath

                    final_alpha_func[function].append(df_final_melt)
  
                
                

                
                
                
    # use not to check if a list is empty            
    if not data_func:
        print("No data loaded.")
        
    
    return data_func, data_layer, final_alpha_func, final_alpha_layer



print("loading data")
df_function, df_layer, df_finals_func, df_finals_layer = load_all_weights(parent_dir)
print("finished loading")



          
summary_stats_func = []
summary_stats_layer = []
          
# loop through all the functions
# layer names are not consistent across all runs, so we will separately loop
# through df_finals_layer for other information
for func_i, dfs in df_finals_func.items():
    # combine all the runs for the function
    df = pd.concat(dfs, ignore_index=True)
    # sort the df
    df = df.sort_values(["experiment", "run_id", "layer"])
    
    # add layer index for plotting since layer names aren't consistent
    df["layer #"] = df.groupby(["experiment", "run_id"]).cumcount()
    
    # plot each experiment, run_id for the function
    plt.figure()
    for (experiment, run_id), group in df.groupby(["experiment", "run_id"]):
        plt.plot(
            group["layer #"],
            group["alpha"],
            marker='.'
            #label=f"{experiment} / {run_id}"
        )
        
    plt.title(f"Final Alphas for {func_i}")
    plt.xlabel("Layer #")
    plt.ylabel("Final Alpha Value")
    #plt.legend(title="Experiment / Run", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{func_i}_final_alphas.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    # finally get some summary statistics per experiment
    grouped = df.groupby("experiment")["alpha"].agg(["mean", "std", "min", "max", "median"]).reset_index()

    for _, row in grouped.iterrows():
        summary_stats_func.append({
            "function": func_i,
            "experiment": row["experiment"],
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "max": row["max"],
            "median": row["median"],
        })
        
    # get summary stats for the whole function
    stats_func = df["alpha"].agg(["mean", "std", "min", "max", "median"])
    summary_stats_func.append({
        "function": func_i,
        "experiment": "all",
        "mean": stats_func["mean"],
        "std": stats_func["std"],
        "min": stats_func["min"],
        "max": stats_func["max"],
        "median": stats_func["median"],
    })
        
        




# loop through all the layers    
for layer_i, dfs in df_finals_layer.items():
    # combine all the runs for the layer
    df = pd.concat(dfs, ignore_index=True)
    
    # make a histogram of the final alphas per function (per layer)
    output_dir = "histograms"
    os.makedirs(output_dir, exist_ok=True)
    for func_i, group in df.groupby("function"):
        plt.figure(figsize = (6,4))  
        # make histogram of chosen bins
        # matplot uses np.hist to the organize the data and then draws the dist.
        bins = 40
        plt.hist(group["alpha"], bins=bins, edgecolor="black")
        plt.title(f"Final Alphas of {layer_i}\nfor {func_i}")
        plt.xlabel("Alpha")
        plt.ylabel(f"Frequency with {bins} bins")
        plt.tight_layout()
        # use dpi=300 for better resolution
        # use bbox_inches="tight" to trim whitespace around image
        file_name = os.path.join(output_dir, f"{func_i}_{layer_i}_histogram.png")
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    
    
    # create a scatter plot of the alpha weights per function
    # overlay a box plot on top the alphas
    # To get the boxplot to align with the scatterplot the numeric positions
    # of the functions must be the same
    
    # get the function names
    func_sorted = sorted(df["function"].unique())
    # get their x position
    func_to_x = {func: i for i, func in enumerate(func_sorted)}
    
    
    # organize the data to use matplotlib's boxplot
    # we need to send an array, where each column is the function's data
    box_data = [df[df["function"] == func]["alpha"].values for func in func_sorted]
  
    
    plt.figure()
    # make the scatter plot
    for (experiment, run_id), group in df.groupby(["experiment", "run_id"]):
        x_vals = group["function"].map(func_to_x)  # Map function names to numeric x
        plt.scatter(
            group["function"],
            group["alpha"],
            marker='.'
            #label=f"{experiment} / {run_id}"
        )
    

    # # overlay the box plot
    # plt.boxplot(box_data, positions=range(len(func_sorted)), widths=0.5)
    # # remake the function names on the x axis
    # plt.xticks(range(len(func_sorted)), func_sorted)#, rotation=45)
        
    plt.title(f"Final Alphas for {layer_i}")
    plt.xlabel("Function")
    plt.ylabel("Final Alpha Value")
    #plt.legend(title="Experiment / Run", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{layer_i}_final_alphas.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    
    
    # finally get some summary statistics
    grouped = df.groupby("function")["alpha"].agg(["mean", "std", "min", "max", "median"]).reset_index()

    for _, row in grouped.iterrows():
        summary_stats_layer.append({
            "Layer": layer_i,
            "function": row["function"],
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "max": row["max"],
            "median": row["median"],
        })
   


