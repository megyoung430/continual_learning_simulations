"""This file specifies all the functions needed to assess the degree of forgetting in
our continual learning task.

To assess forgetting, we analyze the model performance across a series of test angles that
span the entirety of the ring and determine correct choices based on the optimal decision boundary
from a previous task."""

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
    """This function gets the angle used to train Task 0 from the filename of the 
    model.

    Args:
        model_path (pathlib Path object): Path to the network trained on Task 0.

    Raises:
        ValueError: Raised if the filename does not contain Task 0, 
                    indicating the model has not been trained on Task 0.

    Returns:
        int: Returns the angle used to train Task 0.
    """
    filename = model_path.name
    match = re.search(r'Task 0, Theta (\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("The filename does not contain the angle from Task 0.")

def get_task1_angle(model_path):
    """This function gets the angle used to train Task 1 from the filename of the model.

    Args:
        model_path (pathlib Path object): Path to the network trained on Task 1.

    Raises:
        ValueError: Raised if the filename does not contain Task 1, 
                    indicating the model has not been trained on Task 1.

    Returns:
        int: Returns the angle used to train Task 1.
    """
    filename = model_path.name
    match = re.search(r'Task 1, Theta (\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("The filename does not contain the angle from Task 1.")

def get_optimal_decision_boundary(test_theta, task, thetas):
    """This function determines the correct action (right or left) for a given

    Args:
        test_theta (_type_): _description_
        task (_type_): _description_
        thetas (_type_): _description_
    """

    def within_90_degrees(angle_1, angle_2):
        angle_1 = angle_1 % 360
        angle_2 = angle_2 % 360
        difference = (angle_2 - angle_1 + 180) % 360 - 180
        return -90 <= difference <=90
    
    curr_theta = thetas[task]
    # If the difference between the two angles is less than 90 degrees, the optimal action is to
    # choose right
    if within_90_degrees(test_theta, curr_theta):
        return 1
    # Otherwise, choose left
    else:
        return 0

def get_p_train(model_path):
    """_summary_

    Args:
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """_summary_

    Args:
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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

def get_any_auditory_stimulus(test_theta, task, thetas, num_notes=7):
    """_summary_

    Args:
        test_theta (_type_): _description_
        task (_type_): _description_
        thetas (_type_): _description_
        num_notes (int, optional): _description_. Defaults to 7.
    
    Returns:
        stim (num_notes + 1 array):
        correct_choice (int): 
    """
    
    def get_amplitudes(theta, n, order):
        """This function returns the spectrogram of the auditory stimulus.

        Args:
            theta (int): Angle of the sound on the ring.
            n (int): Number of notes used to create the polyphonic sound.
            order (num_notes + 1 array): Optimal order of the frequencies that maximally preserves perceptual uniformity across the ring.

        Returns:
            num_notes + 1 array: Coefficients, representing the power of each frequency, ordered according to the order specified as the input.
        """
        positions = np.linspace(0., 360., n + 1)[:-1]
        positions = np.deg2rad(positions) + np.deg2rad(theta)
        return (0.5 * (np.cos(positions) + 1.))[order]
      
    optimal_order = np.asarray([1, 4, 7, 2, 3, 5, 6]) - 1
    stim = np.ones(shape=num_notes + 1)
    stim[1:] = get_amplitudes(test_theta, num_notes, optimal_order)
    stim = torch.tensor(stim, dtype=torch.float32)
    correct_choice = get_optimal_decision_boundary(test_theta, task, thetas)
    return stim, correct_choice

def assess_forgetting(model_path, num_evaluations=100, training_window=100, theta_window=15):
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
    
    test_thetas = np.arange(0, 361, theta_window)
    points_in_training = np.arange(0, len(data) - 1, training_window)
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
        
        # Evaluate the model trained on Task 1 on the entirety of the ring, 
        # using the optimal decision boundary from Task 0
        for i in range(num_evaluations):
            
            for test_theta in test_thetas:
                # Choose a stimulus, keeping track of the current theta on the ring and the correct choice given an optimal
                # linear decision boundary
                curr_stimulus, correct_choice = get_any_auditory_stimulus(test_theta, task_id, thetas)
                curr_stimulus = curr_stimulus.to(device)
                
                # The output of the network are the Q-values associated with choosing
                # left or right.
                q_values = model(curr_stimulus)
                curr_q_values = q_values.clone().detach().cpu().numpy().copy()
                
                # Then select an action through a softmax function.
                action, action_probabilities = select_action(curr_q_values, beta=data[1]["beta"])
                actions.append(action)
                curr_thetas.append(test_theta)
                correct_choices.append(correct_choice)
        
        # Keep track of the actions made and the correct choice at each point in training
        actions_across_points.append(actions)
        curr_thetas_across_points.append(curr_thetas)
        correct_choices_across_points.append(correct_choices)
        
        # Determine the average performance across all model evaluations
        num_correct = sum(action == correct_choice for action, correct_choice in zip(actions, correct_choices))
        performance = num_correct / num_evaluations
        performances_across_points.append(performance)
    
    return actions_across_points, curr_thetas_across_points, correct_choices_across_points, performances_across_points