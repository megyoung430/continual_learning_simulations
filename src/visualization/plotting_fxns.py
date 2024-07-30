"""This file specifies the functions needed for visualizing."""

import pickle
import matplotlib.pyplot as plt
from ..analysis.task0_analysis_fxns import *
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

def plot_train_accuracy_comparison_across_models(model_names, data_paths_across_models, with_window=False):

    num_models = len(model_names)
    fig, axs = plt.subplots(1, num_models)
    for i in range(num_models):
        
        curr_model = model_names[i]
        data_paths = data_paths_across_models[curr_model]

        if with_window:
            _, _, _, _, mean, std = calculate_train_accuracy_over_training_across_models(data_paths, trial_type="train")
            y_label = "Accuracy over Last 100 Trials"
        else:
            _, mean, std, _, _, _ = calculate_train_accuracy_over_training_across_models(data_paths, trial_type="train")
            y_label = "Accuracy"
        
        axs[i].plot(mean)
        axs[i].fill_between(range(mean.shape[0]), mean - std, mean + std, alpha=0.2)
        axs[i].set_ylabel(y_label)
        axs[i].set_xlabel("Number of Trials")
        axs[i].set_ylim([0,1])
        axs[i].set_xlim([0, mean.shape[0]])
        axs[i].set_title("Model: " + curr_model)
        
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False) 
    
    fig.suptitle("Performance on Train Trials Across Models")
    plt.show()

def plot_validation_accuracy_comparison_across_models(model_names, data_paths_across_models, with_window=False):

    num_models = len(model_names)
    fig, axs = plt.subplots(1, num_models)
    for i in range(num_models):
        
        curr_model = model_names[i]
        data_paths = data_paths_across_models[curr_model]

        if with_window:
            _, _, _, _, mean, std = calculate_validation_accuracy_over_training_across_models(data_paths, trial_type="validation")
            y_label = "Accuracy over Last 100 Trials"
        else:
            _, mean, std, _, _, _ = calculate_validation_accuracy_over_training_across_models(data_paths, trial_type="validation")
            y_label = "Accuracy"
        
        axs[i].plot(mean)
        axs[i].fill_between(range(mean.shape[0]), mean - std, mean + std, alpha=0.2)
        axs[i].set_ylabel(y_label)
        axs[i].set_xlabel("Number of Trials")
        axs[i].set_ylim([0,1])
        axs[i].set_xlim([0, mean.shape[0]])
        axs[i].set_title("Model: " + curr_model)
        
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False) 
    
    fig.suptitle("Performance on Validation Trials Across Models")
    plt.show()

#def plot_forgetting_over_training(model):