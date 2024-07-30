"""This file specifies all the functions needed to assess the degree of forgetting in
our continual learning task."""

import re
import pickle
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch import optim
from ..models.networks import *
from .experiment_fxns import *

def get_task0_angle(model_path):
    filename = model_path.name
    match = re.search(r'Task 0, Theta (\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("The filename does not contain the angle from Task 0.")

def get_task1_angle(model_path):
    filename = model_path.name
    match = re.search(r'Task 1, Theta (\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("The filename does not contain the angle from Task 1.")

def get_optimal_decision_boundary(thetas):
    

def get_p_train(model_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    
    num_trials = len(data) - 1
    num_train_trials = 0
    for i in range(1, num_trials + 1):
        curr_trial_info = data[i]
        if curr_trial_info["trial_type"] == "train":
            num_train_trials = num_train_trials + 1
    return num_train_trials/num_trials

def get_p_test(model_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    
    num_trials = len(data) - 1
    num_test_trials = 0
    for i in range(1, num_trials + 1):
        curr_trial_info = data[i]
        if curr_trial_info["trial_type"] == "test":
            num_test_trials = num_test_trials + 1
    return num_test_trials/num_trials

def get_p_validation(model_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    
    num_trials = len(data) - 1
    num_validation_trials = 0
    for i in range(1, num_trials + 1):
        curr_trial_info = data[i]
        if curr_trial_info["trial_type"] == "validation":
            num_validation_trials = num_validation_trials + 1
    return num_validation_trials/num_trials

def get_p_reward_train(model_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    
    num_trials = len(data) - 1
    num_correct_train_trials = 0
    num_rewarded_correct_train_trials = 0
    for i in range(1, num_trials + 1):
        curr_trial_info = data[i]
        if curr_trial_info["trial_type"] == "train":
            if curr_trial_info["action"] == curr_trial_info["correct_choice"]:
                num_correct_train_trials = num_correct_train_trials + 1
                if curr_trial_info["reward"] == 1:
                    num_rewarded_correct_train_trials = num_rewarded_correct_train_trials + 1
    return num_rewarded_correct_train_trials/num_correct_train_trials

def get_p_reward_test_validation(model_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    
    num_trials = len(data) - 1
    num_test_validation_trials = 0
    num_rewarded_test_validation_trials = 0
    for i in range(1, num_trials + 1):
        curr_trial_info = data[i]
        if curr_trial_info["trial_type"] == "test" or curr_trial_info["trial_type"] == "validation":
                num_test_validation_trials = num_test_validation_trials + 1
                if curr_trial_info["reward"] == 1:
                    num_rewarded_test_validation_trials = num_rewarded_test_validation_trials + 1
    return num_rewarded_test_validation_trials/num_test_validation_trials

def assess_forgetting(model_path, num_evaluations=1000, window=100):
    """This function calculates the performance on task 0 at different periods in training on task 1.

    Args:
        model_path (pathlib Path object): _description_
        num_evaluations (int, optional): _description_. Defaults to 1000.
        window (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """

    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    
    # Get the relevant task info
    trial_type = "train"
    task_id = 0
    theta_0 = get_task0_angle(model_path)
    theta_1 = get_task1_angle(model_path)
    thetas = [theta_0, theta_1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points_in_training = np.arange(0, len(data), window)
    actions_across_points = []
    curr_thetas_across_points = []
    correct_choices_across_points = []
    performances_across_points = []
    for point_in_training in points_in_training:
        actions = []
        curr_thetas = []
        correct_choices = []

        model = data[point_in_training + 1]["model"]
        model.to(device)
        model.eval()
        
        # Evaluate the model trained on Task 1 on Task 0
        for i in range(num_evaluations):
            # Choose a stimulus, keeping track of the current theta on the ring and the correct choice given an optimal
            # linear decision boundary
            curr_stimulus, curr_theta, correct_choice = get_auditory_stimulus(trial_type, task_id, thetas)
            curr_stimulus = curr_stimulus.to(device)
            
            # The output of the network are the Q-values associated with choosing
            # left or right.
            q_values = model(curr_stimulus)
            curr_q_values = q_values.clone().detach().cpu().numpy().copy()
            
            # Then select an action through a softmax function.
            action, action_probabilities = select_action(curr_q_values, beta=data[1]["beta"])
            actions.append(action)
            curr_thetas.append(curr_theta)
            correct_choices.append(correct_choice)
        
        # Keep track of the actions made and the correct choice
        actions_across_points.append(actions)
        curr_thetas_across_points.append(curr_thetas)
        correct_choices_across_points.append(correct_choices)
        
        # Determine the average performance across all model evaluations
        num_correct = sum(action == correct_choice for action, correct_choice in zip(actions, correct_choices))
        performance = num_correct / num_evaluations
        performances_across_points.append(performance)
    
    return actions_across_points, curr_thetas_across_points, correct_choices_across_points, performances_across_points