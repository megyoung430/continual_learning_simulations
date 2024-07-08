"""This file specifies the functions needed to analyze the trained models and 
their learning trajectories."""

import pickle
import numpy as np
from ..models.networks import *

def calculate_accuracy_over_training(data_path, window=100):
    """This function calculates the accuracy of the model over the course of training.

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
        correct_choice_history.append(curr_trial_info["action"] == curr_trial_info["correct_choice"])
        if i > window:
            running_accuracy_over_training.append(np.sum(correct_choice_history[-window:])/window)
    accuracy_over_training = np.cumsum(correct_choice_history)/np.arange(1, num_trials + 1, 1)
    return accuracy_over_training, running_accuracy_over_training

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