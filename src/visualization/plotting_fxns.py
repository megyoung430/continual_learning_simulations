"""This file specifies the functions needed for visualizing."""

import pickle
import matplotlib.pyplot as plt
from ..simulations.analysis_fxns import calculate_accuracy_over_training, get_loss_over_training
from ..models.networks import *

def plot_summary_figure(data_path, save_path):
    """_summary_

    Args:
        data_path (_type_): _description_
        data_path (_type_): _description_
    """

    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    curr_model = data[0]["model"]

    fig, axs = plt.subplots(1, 2)
    if type(curr_model) is DeepRLAuditoryDiscriminationNetwork and curr_model.rpe_type == "partial":
        accuracy_over_training, running_accuracy_over_training = calculate_accuracy_over_training(data_path)
        loss_l1_over_training, loss_l2_const_over_training, loss_l2_stim_over_training = get_loss_over_training(data_path)
        
        axs[0].plot(running_accuracy_over_training)
        axs[1].plot(loss_l1_over_training, label="W1 Loss")
        axs[1].plot(loss_l2_const_over_training, label="W2_Const Loss")
        axs[1].plot(loss_l2_stim_over_training, label="W2_Stim Loss")
        axs[1].legend()
    else:
        accuracy_over_training, running_accuracy_over_training = calculate_accuracy_over_training(data_path)
        loss_over_training = get_loss_over_training(data_path)
        
        axs[0].plot(running_accuracy_over_training)
        axs[1].plot(loss_over_training)
    
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Number of Trials")

    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Number of Trials")

    plt.show()