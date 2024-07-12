"""This file specifies the functions needed to analyze the trained models and 
their learning trajectories."""

import pickle
import glob
from pathlib import Path
import numpy as np
from ..models.networks import *

def get_all_models(data_directory, model_name):

    pattern = f"*{model_name}*.pk1"
    return list(data_directory.glob(pattern))

def calculate_accuracy_over_training(data_path, trial_type, window=100):
    """This function calculates the accuracy of the model on train trials over the course of training.

    Args:
        data_path (Path object): Path to the data.

    Returns:
        accuracy_over_training (num_trials array): Running accuracy of the model after each trial.
    """

    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    num_trials = len(data) - 1
    correct_choice_history = []
    running_accuracy_over_training = []
    for i in range(1, num_trials + 1):
        curr_trial_info = data[i]
        if trial_type == "all":
            correct_choice_history.append(curr_trial_info["action"] == curr_trial_info["correct_choice"])
            if i > window:
                running_accuracy_over_training.append(np.sum(correct_choice_history[-window:])/window)
        else:
            if curr_trial_info["trial_type"] == trial_type:
                correct_choice_history.append(curr_trial_info["action"] == curr_trial_info["correct_choice"])
                if i > window:
                    running_accuracy_over_training.append(np.sum(correct_choice_history[-window:])/window)
    accuracy_over_training = np.cumsum(correct_choice_history)/np.arange(1, len(correct_choice_history) + 1, 1)
    return accuracy_over_training, running_accuracy_over_training

def pad_arrays(arrays, pad_value=np.nan):
    """Pads arrays to the same length with the specified pad value.

    Args:
        arrays (_type_): _description_
        pad_value (_type_, optional): _description_. Defaults to np.nan.

    Returns:
        _type_: _description_
    """
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), constant_values=pad_value) for arr in arrays]
    return np.array(padded_arrays)

def calculate_accuracy_over_training_across_models(data_paths, trial_type, window=100):
    """_summary_

    Args:
        data_paths (_type_): _description_
        window (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    accuracies_over_training = []
    running_accuracies_over_training = []
    for data_path in data_paths:
        accuracy_over_training, running_accuracy_over_training = calculate_accuracy_over_training(data_path, trial_type, window=window)
        accuracies_over_training.append(accuracy_over_training)
        running_accuracies_over_training.append(running_accuracy_over_training)
    
    padded_accuracies = pad_arrays(accuracies_over_training)
    padded_running_accuracies = pad_arrays(running_accuracies_over_training)
    
    mean_accuracies_over_training = np.nanmean(padded_accuracies, axis=0)
    std_accuracies_over_training = np.nanstd(padded_accuracies, axis=0)
    mean_running_accuracies_over_training = np.nanmean(padded_running_accuracies, axis=0)
    std_running_accuracies_over_training = np.nanstd(padded_running_accuracies, axis=0)
    
    return (accuracies_over_training, mean_accuracies_over_training, std_accuracies_over_training,
            running_accuracies_over_training, mean_running_accuracies_over_training, std_running_accuracies_over_training)

def get_loss_over_training(data_path):
    """_summary_

    Args:
        data_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    num_trials = len(data) - 1
    curr_model = data[0]["model"]
    if type(curr_model) is DeepRLAuditoryDiscriminationNetwork and curr_model.rpe_type == "partial":
        loss_l1_over_training = [data[i]["loss_l1"] for i in range(1, num_trials + 1)]
        loss_l2_const_over_training = [data[i]["loss_l2_const"] for i in range(1, num_trials + 1)]
        loss_l2_stim_over_training = [data[i]["loss_l2_stim"] for i in range(1, num_trials + 1)]
        return loss_l1_over_training, loss_l2_const_over_training, loss_l2_stim_over_training
    else:
        loss_over_training = [data[i]["loss"] for i in range(num_trials)]
        return loss_over_training