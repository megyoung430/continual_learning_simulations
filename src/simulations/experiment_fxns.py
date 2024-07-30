"""This file specifies all the functions needed to run a complete simulation of 
our continual learning task."""

import random
import pickle
import numpy as np
import torch
from torch import nn
from torch import optim
from ..models.networks import *

def get_trial_type(p_train=0.8, p_test=0.5):
    """This function determines the type of the upcoming trial.

    Args:
        p_train (float, optional): Probability of a train trial (vs. test/validation trial). Defaults to 0.8.
        p_test (float, optional): If not a train trial, probability of a test trial (vs. validation trial). Defaults to 0.5.

    Returns:
        string: Trial type of the upcoming trial. Either "train", "test", or "validation".
    """
    train = random.random()
    if train < p_train:
        return "train"
    else:
        test = random.random()
        if test < p_test:
            return "test"
        else:
            return "validation"

def get_general_stimulus(p_stim_right=0.5):
    """This function returns the stimulus/input to the network for a given trial in the general task.

    Args:
        p_stim_right (float, optional): Probability that the stimulus will be that associated with a right choice. Defaults to 0.5.

    Returns:
        stim (num_notes + 1 tensor): The first element is a constant bias term; the second, the presence of a right stimulus
                                     (1 if true; 0 if false); the third, the presence of a left stimulus (1 if true; 0 if false).
    """
    stim_right = random.random()
    if stim_right < p_stim_right:
        stim = torch.tensor([1,1,0], dtype=torch.float32)
        correct_choice = 1
    else:
        stim = torch.tensor([1,0,1], dtype=torch.float32)
        correct_choice = 0
    return stim, correct_choice

def get_auditory_stimulus(trial_type, task_id=0, thetas=[0,90], p_stim_right=0.5, num_notes=7):
    """This function returns the stimulus / input to the network for a given trial in the auditory task.

    Args:
        trial_type (string): Type of the current trial. Either "train", "test", or "validation".
        task_id (int, optional): Number of the current task. Either 0 for task 1 or 1 for task 2. Defaults to 0.
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        p_stim_right (float, optional): Probability that the stimulus will be that associated with a right choice. Defaults to 0.5.
        num_notes (int, optional): Number of notes used to create the polyphonic sound. Defaults to 7.

    Returns:
        stim (num_notes + 1 tensor): The first element is a constant bias term; the remaining num_notes elements are the
                                     amplitudes of the notes for the sound stimulus (i.e., the spectrogram of the stimulus).
        curr_theta (float): Angle for the current task.
        correct_choice (int): Correct action associated with the current stimulus. Either 0 (left choice) or 1 (right choice).
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

    if trial_type == "train":
        stim_right = random.random()
        if stim_right < p_stim_right:
            stim = np.ones(shape=num_notes + 1)
            stim[1:] = get_amplitudes(thetas[task_id], num_notes, optimal_order)
            stim = torch.tensor(stim, dtype=torch.float32)
            curr_theta = thetas[task_id]
            correct_choice = 1
        else:
            stim = np.ones(shape=num_notes + 1)
            stim[1:] = get_amplitudes(thetas[task_id] + 180, num_notes, optimal_order)
            stim = torch.tensor(stim, dtype=torch.float32)
            curr_theta = thetas[task_id] + 180
            correct_choice = 0
    elif trial_type == "validation":
        stim_right = random.random()
        if stim_right < p_stim_right:
            stim = np.ones(shape=num_notes + 1)
            stim[1:] = get_amplitudes(thetas[0], num_notes, optimal_order)
            stim = torch.tensor(stim, dtype=torch.float32)
            curr_theta = thetas[0]
            correct_choice = 1
        else:
            stim = np.ones(shape=num_notes + 1)
            stim[1:] = get_amplitudes(thetas[0] + 180, num_notes, optimal_order)
            stim = torch.tensor(stim, dtype=torch.float32)
            curr_theta = thetas[0] + 180
            correct_choice = 0
    elif trial_type == "test":
        curr_theta = random.random()*360
        stim = np.ones(shape=num_notes + 1)
        stim[1:] = get_amplitudes(curr_theta, num_notes, optimal_order)
        stim = torch.tensor(stim, dtype=torch.float32)
        if np.abs(curr_theta - thetas[task_id]) <= 90:
            correct_choice = 1
        else:
            correct_choice = 0
    return stim, curr_theta, correct_choice

def auditory_sensitivity_function(stim):
    """This function implements a sensitivity function, intended to reflect differences in perceptual
    sensitivity to different frequencies.

    Args:
        stim (num_notes + 1 tensor): The first element is a constant bias term; the remaining num_notes elements are the
                                     amplitudes of the notes for the sound stimulus (i.e., the spectrogram of the stimulus).

    Returns:
        stim (num_notes + 1 tensor): The first element is a constant bias term; the remaining num_notes elements are the 
                                     perceived amplitudes of the notes for the sound stimulus.
    """
    return stim

def select_action(q_values, beta=2.5):
    """This function implements softmax action selection.

    Args:
        q_values (_type_): Action values.
        beta (float, optional): Inverse temperature parameter. Defaults to 2.5.

    Returns:
        action (int): Chosen action. Either 0 (left choice) or 1 (right choice) or 2 (inaction) for networks that can choose not to act.
        probabilities (num_actions array): Probability of choosing either 0 (left choice) or 1 (right choice) or 2 (inaction) for
                                            networks that can choose not to act.
    """
    exponent = np.exp(beta * q_values)
    probabilities = exponent / np.sum(exponent)
    action = np.random.choice(range(len(q_values)), p=probabilities)
    return action, probabilities

def get_general_reward(trial_type, action, correct_choice, p_reward_train=1, p_reward_test_validation=0.5,
                        with_inaction=False, reward_volume=20, action_penalty=5):
    """This function determines the reward delivered on a given trial in the general version of the task.

    Args:
        trial_type (string): The type of the current trial. Either train, test, or validation.
        action (int): Action the agent chose. Either 0 (left choice) or 1 (right choice) or 2 (inaction) for networks that can choose not to act.
        correct_choice (int): Correct action associated with the stimulus. Either 0 (left choice) or 1 (right choice).
        p_reward_train (float, optional): Probability that the current trial is rewarded for train trials. Defaults to 1.
        p_reward_test_validation (float, optional): Probability that the current trial is rewarded for validation and test trials. Defaults to 0.5.
        with_inaction (bool, optional): If true, indicates that the model can also choose inaction. Defaults to False.
        reward_volume (int, optional): Only relevant if with_inaction is True. Amount of reward delivered. Defaults to 20.
        action_penalty (int, optional): Only relevant if with_inaction is True. Cost of acting. Defaults to 5.

    Returns:
        int: Indicates presence or absence of reward. If with_inaction is True, indicates net amount of reward (reward_volume - action_penalty).
    """

    if with_inaction:
        net_reward = reward_volume - action_penalty 
        # If the network chooses not to act, then there's neither a reward nor an action penalty
        if action == 2:
            return 0
    else:
        net_reward = 1

    if trial_type == "train":
        if action == correct_choice:
            reward = random.random()
            if reward < p_reward_train:
                return net_reward
            else:
                return 0
        else:
            return 0
    else:
        reward = random.random()
        if reward < p_reward_test_validation:
            return net_reward
        else:
            return 0

def get_auditory_reward(trial_type, curr_theta, action, task_id=0, thetas=[0,90], p_reward_train=1, p_reward_test_validation=0.5,
                         with_inaction=False, reward_volume=20, action_penalty=5):
    """This function determines the reward delivered on a given trial in the auditory version of the task.

    Args:
        trial_type (string): The type of the current trial. Either train, test, or validation.
        curr_theta (float): The theta used to generate the spectrogram for the stimulus for the current trial.
        action (int): Action the agent chose. Either 0 (left choice) or 1 (right choice) or 2 (inaction) for networks that can choose not to act.
        task_id (int, optional): The number of the task. Either 0 (task 1) or 1 (task 2).
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        p_reward_train (float, optional): Probability that the current trial is rewarded for train trials. Defaults to 1.
        p_reward_test_validation (float, optional): Probability that the current trial is rewarded for validation and test trials. Defaults to 0.5.
        with_inaction (bool, optional): If true, indicates that the model can also choose inaction. Defaults to False.
        reward_volume (int, optional): Only relevant if with_inaction is True. Amount of reward delivered. Defaults to 20.
        action_penalty (int, optional): Only relevant if with_inaction is True. Cost of acting. Defaults to 5.

    Returns:
        int: Indicates presence or absence of reward. If with_inaction is True, indicates net amount of reward (reward_volume - action_penalty).
    """

    if with_inaction:
        net_reward = reward_volume - action_penalty
        # If the network chooses not to act, then there's neither a reward nor an action penalty
        if action == 2:
            return 0
    else:
        net_reward = 1

    if trial_type == "train":
        # If the current angle is the first sound of the task, then reward the right choice
        if curr_theta == thetas[task_id]:
            if action == 1:
                reward = random.random()
                if reward < p_reward_train:
                    return net_reward
                else:
                    return 0
            else:
                return 0
        # Otherwise, reward left choice
        else:
            if action == 0:
                reward = random.random()
                if reward < p_reward_train:
                    return net_reward
                else:
                    return 0
            else:
                return 0
    else:
        reward = random.random()
        if reward < p_reward_test_validation:
            return net_reward
        else:
            return 0

def run_shallow_rl_experiment(spectrogram=True, task_id=0, thetas=[0,90], model_path=None, num_notes=7, p_train=0.8, 
                                num_trials=10000, learning_rate=0.1, beta=2.5, rpe_type="full", save_data=True, save_path=None):
    """This function runs an experiment similar to that used to train the animals for a shallow network trained via reinforcement learning.

    Args:
        spectrogram (bool, optional): If true, then the network trains on the full auditory task; if false, the network trains on the simpler, general task. Defaults to True.
        task_id (int, optional): The number of the task. Either 0 (task 1) or 1 (task 2). Defaults to 0.
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        model_path (pathlib Path object): Path to the previously trained network. Needed if task_id is not zero. Defaults to None.
        num_notes (int, optional): Number of notes used to create the polyphonic sound. Defaults to 7.
        p_train (float, optional): Probability of a train trial (vs. test/validation trial). Defaults to 0.8.
        num_trials (int, optional): Number of trials in the experiment. Defaults to 10000.
        learning_rate (float, optional): Learning rate for the network. Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter for the softmax action selection. Defaults to 2.5.
        rpe_type (str, optional): Specifies the type of the RPE signal, either "full" or "partial". Defaults to "full".
        save_data (bool, optional): If true, after every iteration, this function saves a dictionary with relevant trial variables. Defaults to True.
        save_path (pathlib Path object): Path to where data should be saved. Defaults to None.
    """
    # If training on the basic task (input indicates just the presence of the left or right stimulus), then the input dimension
    # (parametrized in the networks as as num_notes) is 2.
    if not spectrogram:
        num_notes = 2
    
    # If this is the first task, then initialize a new network
    if task_id == 0:
        model = ShallowRLAuditoryDiscriminationNetwork(rpe_type=rpe_type, num_notes=num_notes)
    # However, if this is one of the later tasks, load a previously trained network
    else:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        model = data[-1]["model"]
        # Check that the model is the correct type
        assert(type(model) == ShallowRLAuditoryDiscriminationNetwork)
        assert(model.rpe_type == rpe_type)
    
    # Check to see if GPU is available; otherwise, use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Start by keeping track of the initialized model
    if task_id == 0:
        data = [{
            "model": model
        }]
    else:
        data = [{
            "model": model,
            "model_path": model_path
        }]
    
    if model.rpe_type == "full":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif model.rpe_type == "partial":
        optimizer = optim.SGD([
            {'params': [model.l1_weights_const]},
            {'params': [model.l1_weights_stim]}
        ], lr=learning_rate)
    criterion = nn.MSELoss()
    
    # In the reinforcement version of the model, the output of the network reflects the Q-values associated with choosing left of right,
    # which are updated based on the RPE.
    for i in range(num_trials):
        # Determine the trial type, either train, test, or validation
        trial_type = get_trial_type(p_train=p_train)

        # Choose a stimulus, keeping track of the current theta on the ring and the correct choice given an optimal
        # linear decision boundary
        if spectrogram:
            curr_stimulus, curr_theta, correct_choice = get_auditory_stimulus(trial_type, task_id, thetas)
        else:
            curr_stimulus, correct_choice = get_general_stimulus()
        curr_stimulus = curr_stimulus.to(device)
        
        # The output of the network are the Q-values associated with choosing
        # left or right.
        q_values = model(curr_stimulus)
        curr_q_values = q_values.clone().detach().cpu().numpy().copy()
        
        # Then select an action through a softmax function.
        action, action_probabilities = select_action(curr_q_values, beta=beta)
        
        # Then determine if the choice is rewarded
        if spectrogram:
            reward = get_auditory_reward(trial_type, curr_theta, action, task_id, thetas)
        else:
            reward = get_general_reward(trial_type, action, correct_choice)
        
        # If we're using the full RPE, we update all the weights based on the same loss function
        if model.rpe_type == "full":
            # Update the relevant Q-value based on the RPE
            target_q_values = curr_q_values.copy()
            target_q_values[action] = curr_q_values[action] + (reward - curr_q_values[action])
            target_q_values = torch.tensor(target_q_values, dtype=torch.float32).to(device)
            loss = criterion(q_values, target_q_values)
            assert np.isclose(loss.item(), 0.5*(reward - curr_q_values[action])**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if spectrogram:
                trial_data = {
                    "model": model,
                    "loss": loss.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "curr_theta": curr_theta,
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward
                }
            else:
                trial_data = {
                    "model": model,
                    "loss": loss.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward,
                }
            data.append(trial_data)
        
        # If we're using partial RPEs, we update W1_const and W1_stim independently, using three
        # different loss functions
        elif model.rpe_type == "partial":
            old_w1_const = model.l1_weights_const.clone().detach().cpu().numpy().copy()
            old_w1_stim = model.l1_weights_stim.clone().detach().cpu().numpy().copy()

            w1_const = model.l1_weights_const.clone().detach().cpu().numpy().copy()
            w1_stim = model.l1_weights_stim.clone().detach().cpu().numpy().copy()
            w1 = np.concatenate((w1_const, w1_stim), axis=1)

            # Compute the Q-values using only the const term
            const_term_input = curr_stimulus.clone()
            const_term_input[1:] = 0  # Zero out stimulus terms
            const_term_input.to(device)
            const_q_values = model(const_term_input)
            expected_const_q_values = w1[0]
            assert np.allclose(const_q_values.clone().detach().cpu().numpy().copy(), expected_const_q_values, atol=1e-03), f"Expected: {expected_const_q_values}, Got: {const_q_values.clone().detach().cpu().numpy().copy()}"

            # Compute the Q-values using only the stim term
            stim_term_input = curr_stimulus.clone()
            stim_term_input[0] = 0  # Zero out the constant term
            stim_term_input.to(device)
            stim_q_values = model(stim_term_input)
            expected_stim_q_values = np.zeros(shape=stim_q_values.clone().detach().cpu().numpy().copy().shape)
            for i in range(1, num_notes + 1):
                expected_stim_q_values = expected_stim_q_values + curr_stimulus[i].clone().detach().cpu().numpy().copy() * w1[i]
            assert np.allclose(stim_q_values.clone().detach().cpu().numpy().copy(), expected_stim_q_values, atol=1e-03), f"Expected: {expected_stim_q_values}, Got: {stim_q_values.clone().detach().cpu().numpy().copy()}"

            # Calculate the loss for W1_const
            curr_const_q_values = const_q_values.clone().detach().cpu().numpy().copy()
            const_corticostriatal_loss = reward - curr_const_q_values[action]
            target_const_q_values = curr_const_q_values.copy()
            target_const_q_values[action] = target_const_q_values[action] + const_corticostriatal_loss
            target_const_q_values = torch.tensor(target_const_q_values, dtype=torch.float32).to(device)

            loss_l1_const = criterion(const_q_values, target_const_q_values)
            expected_loss_l1_const = 0.5 * (reward - curr_const_q_values[action]) ** 2
            assert np.isclose(loss_l1_const.item(), expected_loss_l1_const, atol=1e-03), f"Expected: {expected_loss_l1_const}, Got: {loss_l1_const.item()}"

            # Update W1_const
            optimizer.zero_grad()
            model.l1_weights_const.requires_grad = True
            model.l1_weights_stim.requires_grad = False
            loss_l1_const.backward(retain_graph=True)
            optimizer.step()

            # Calculate the loss for W1_stim
            curr_stim_q_values = stim_q_values.clone().detach().cpu().numpy().copy()
            stim_corticostriatal_loss = reward - curr_stim_q_values[action]
            target_stim_q_values = curr_stim_q_values.copy()
            target_stim_q_values[action] = target_stim_q_values[action] + stim_corticostriatal_loss
            target_stim_q_values = torch.tensor(target_stim_q_values, dtype=torch.float32).to(device)

            loss_l1_stim = criterion(stim_q_values, target_stim_q_values)
            expected_loss_l1_stim = 0.5 * (reward - curr_stim_q_values[action]) ** 2
            assert np.isclose(loss_l1_stim.item(), expected_loss_l1_stim, atol=1e-03), f"Expected: {expected_loss_l1_stim}, Got: {loss_l1_stim.item()}"

            # Update W1_stim
            optimizer.zero_grad()
            model.l1_weights_const.requires_grad = False
            model.l1_weights_stim.requires_grad = True
            loss_l1_stim.backward()
            optimizer.step()

            # Need to unfreeze all the weights again
            model.l1_weights_const.requires_grad = True
            model.l1_weights_stim.requires_grad = True

            new_w1_const = model.l1_weights_const.clone().detach().cpu().numpy().copy()
            new_w1_stim = model.l1_weights_stim.clone().detach().cpu().numpy().copy()

            delta_w1_const = new_w1_const - old_w1_const
            delta_w1_stim = new_w1_stim - old_w1_stim

            expected_delta_w1_const = np.zeros(shape=delta_w1_const.shape)
            expected_delta_w1_stim = np.zeros(shape=delta_w1_stim.shape)

            for i in range(1, num_notes + 1):
                expected_delta_w1_stim[action, i - 1] = curr_stimulus[i] * w1[i]

            expected_delta_w1_const[action] = learning_rate * const_corticostriatal_loss * w1[0]
            expected_delta_w1_stim = learning_rate * stim_corticostriatal_loss * expected_delta_w1_stim

            assert np.allclose(delta_w1_const, expected_delta_w1_const, atol=1e-03), f"Expected: {expected_delta_w1_const}, Got: {delta_w1_const}"
            assert np.allclose(delta_w1_stim, expected_delta_w1_stim, atol=1e-03), f"Expected: {expected_delta_w1_stim}, Got: {delta_w1_stim}"

            if spectrogram:
                trial_data = {
                    "model": model,
                    "loss_l1_const": loss_l1_const.item(),
                    "loss_l1_stim": loss_l1_stim.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "curr_theta": curr_theta,
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward
                }
            else:
                trial_data = {
                    "model": model,
                    "loss_l1_const": loss_l1_const.item(),
                    "loss_l1_stim": loss_l1_stim.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward
                }
            data.append(trial_data)
    
    if save_data:
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

def run_deep_rl_with_inaction_experiment(spectrogram=True, task_id=0, thetas=[0,90], model_path=None, num_notes=7, p_train=0.8, p_reward_train=1, p_reward_test_validation=0.5,
                                          num_trials=10000, learning_rate=0.1, beta=2.5, reward_volume=20, action_penalty=5, 
                                          rpe_type="full", tonotopy=False, save_data=True, save_path=None):
    """This function runs an experiment similar to that used to train the animals for a deep network trained via reinforcement learning.

    Args:
        spectrogram (bool, optional): If true, then the network trains on the full auditory task; if false, the network trains on the simpler, general task. Defaults to True.
        task_id (int, optional): The number of the task. Either 0 (task 1) or 1 (task 2). Defaults to 0.
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        model_path (pathlib Path object): Path to the previously trained network. Needed if task_id is not zero. Defaults to None.
        num_notes (int, optional): Number of notes used to create the polyphonic sound. Defaults to 7.
        p_train (float, optional): Probability of a train trial (vs. test/validation trial). Defaults to 0.8.
        p_reward_train (float, optional): Probability that the current trial is rewarded for train trials. Defaults to 1.
        p_reward_test_validation (float, optional): Probability that the current trial is rewarded for validation and test trials. Defaults to 0.5.
        num_trials (int, optional): Number of trials in the experiment. Defaults to 10000.
        learning_rate (float, optional): Learning rate for the network. Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter for the softmax action selection. Defaults to 2.5.
        reward_volume (int, optional): Amount of reward delivered. Defaults to 20.
        action_penalty (int, optional): Cost of acting. Defaults to 5.
        rpe_type (str, optional): Specifies the type of the RPE signal, either "full" or "partial". Defaults to "full".
        tonotopy (bool, optional): If true, then the first layer weights are diagonal, motivated by the existence of tonotopy in auditory cortex. Defaults to False.
        save_data (bool, optional): If true, after every iteration, this function saves a dictionary with relevant trial variables. Defaults to True.
        save_path (pathlib Path object): Path to where data should be saved. Defaults to None.
    """
    # If training on the basic task (input indicates just the presence of the left or right stimulus), then the input dimension
    # (parametrized in the networks as as num_notes) is 2.
    if not spectrogram:
        num_notes = 2

    # If this is the first task, then initialize a new network
    if task_id == 0:
        model = DeepRLAuditoryDiscriminationNetwork(rpe_type=rpe_type, tonotopy=tonotopy, num_notes=num_notes, num_actions=3)
    # However, if this is one of the later tasks, load a previously trained network
    else:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        model = data[-1]["model"]
        # Check that the model is the correct type
        assert(type(model) == DeepRLAuditoryDiscriminationNetwork)
        assert(model.rpe_type == rpe_type)
        assert(model.tonotopy == tonotopy)
    
    # Check to see if GPU is available; otherwise, use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Start by keeping track of the initialized model
    if task_id == 0:
        data = [{
            "model": model
        }]
    else:
        data = [{
            "model": model,
            "model_path": model_path
        }]
    
    if model.rpe_type == "full":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif model.rpe_type == "partial":
        optimizer = optim.SGD([
            {'params': [model.l1_weights] if model.tonotopy else model.l1_weights.parameters()},
            {'params': [model.l2_weights_const]},
            {'params': [model.l2_weights_stim]}
        ], lr=learning_rate)
    criterion = nn.MSELoss()
    
    # In the reinforcement version of the model, the output of the network reflects the Q-values associated with choosing left of right,
    # which are updated based on the RPE.
    for i in range(num_trials):
        # Determine the trial type, either train, test, or validation
        trial_type = get_trial_type(p_train=p_train)

        # Choose a stimulus, keeping track of the current theta on the ring and the correct choice given an optimal
        # linear decision boundary
        if spectrogram:
            curr_stimulus, curr_theta, correct_choice = get_auditory_stimulus(trial_type, task_id, thetas)
        else:
            curr_stimulus, correct_choice = get_general_stimulus()
        curr_stimulus = curr_stimulus.to(device)
        
        # The output of the network are the Q-values associated with choosing
        # left or right.
        q_values = model(curr_stimulus)
        curr_q_values = q_values.clone().detach().cpu().numpy().copy()
        
        # Then select an action through a softmax function.
        action, action_probabilities = select_action(curr_q_values, beta=beta)
        
        # Then determine if the choice is rewarded
        if spectrogram:
            reward = get_auditory_reward(trial_type, curr_theta, action, task_id, thetas, 
                                          p_reward_train=p_reward_train, p_reward_test_validation=p_reward_test_validation,
                                          with_inaction=True, reward_volume=reward_volume, action_penalty=action_penalty)
        else:
            reward = get_general_reward(trial_type, action, correct_choice, 
                                         p_reward_train=p_reward_train, p_reward_test_validation=p_reward_test_validation,
                                         with_inaction=True, reward_volume=reward_volume, action_penalty=action_penalty)
        
        # If we're using the full RPE, we update all the weights based on the same loss function
        if model.rpe_type == "full":
            # Update the relevant Q-value based on the RPE
            target_q_values = curr_q_values.copy()
            target_q_values[action] = curr_q_values[action] + (reward - curr_q_values[action])
            target_q_values = torch.tensor(target_q_values, dtype=torch.float32).to(device)
            loss = criterion(q_values, target_q_values)
            assert np.isclose(loss.item(), 1/3*(reward - curr_q_values[action])**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if spectrogram:
                trial_data = {
                    "model": model,
                    "loss": loss.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "curr_theta": curr_theta,
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward,
                    "reward_volume": reward_volume,
                    "action_penalty": action_penalty
                }
            else:
                trial_data = {
                    "model": model,
                    "loss": loss.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward,
                    "reward_volume": reward_volume,
                    "action_penalty": action_penalty
                }
            data.append(trial_data)

        # If we're using partial RPEs, we update W1, W2_const, and W2_stim independently, using three
        # different loss functions
        elif model.rpe_type == "partial":
            
            if tonotopy:
                old_w1 = model.l1_weights.clone().detach().cpu().numpy().copy()
                old_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                old_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()
                old_w2 = np.concatenate((old_w2_const, old_w2_stim), axis=1)
            else:
                old_w1 = model.l1_weights.weight.clone().detach().cpu().numpy().copy()
                old_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                old_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()
                old_w2 = np.concatenate((old_w2_const, old_w2_stim), axis=1)

            # Compute the Q-values using only the const term
            const_term_input = curr_stimulus.clone()
            const_term_input[1:] = 0  # Zero out stimulus terms
            const_term_input.to(device)
            const_q_values = model(const_term_input)
            if tonotopy:
                expected_const_q_values = old_w1[0] * old_w2[:,0]
            else:
                expected_const_q_values = old_w2 @ old_w1[:,0]
            assert np.allclose(const_q_values.clone().detach().cpu().numpy().copy(), expected_const_q_values, atol=1e-03), f"Expected: {expected_const_q_values}, Got: {const_q_values.clone().detach().cpu().numpy().copy()}"

            # Compute the Q-values using only the stim term
            stim_term_input = curr_stimulus.clone()
            stim_term_input[0] = 0  # Zero out the constant term
            stim_term_input.to(device)
            stim_q_values = model(stim_term_input)
            expected_stim_q_values = np.zeros(shape=stim_q_values.clone().detach().cpu().numpy().copy().shape)
            for i in range(1, num_notes + 1):
                if tonotopy:
                    expected_stim_q_values = expected_stim_q_values + curr_stimulus[i].clone().detach().cpu().numpy().copy() * old_w1[i] * old_w2[:,i]
                else:
                    expected_stim_q_values = expected_stim_q_values + curr_stimulus[i].clone().detach().cpu().numpy().copy() * old_w2 @ old_w1[:,i]
            assert np.allclose(stim_q_values.clone().detach().cpu().numpy().copy(), expected_stim_q_values, atol=1e-03), f"Expected: {expected_stim_q_values}, Got: {stim_q_values.clone().detach().cpu().numpy().copy()}"

            # Calculate the loss for W1
            cortical_loss = reward - curr_q_values[action]
            target_cortical_q_values = curr_q_values.copy()
            target_cortical_q_values[action] = curr_q_values[action] + cortical_loss
            target_cortical_q_values = torch.tensor(target_cortical_q_values, dtype=torch.float32).to(device)

            loss_l1 = criterion(q_values, target_cortical_q_values)
            expected_loss_l1 = 1/3 * (cortical_loss) ** 2
            assert np.isclose(loss_l1.item(), expected_loss_l1, atol=1e-03), f"Expected: {expected_loss_l1}, Got: {loss_l1.item()}"

            # Update W1
            optimizer.zero_grad()
            if tonotopy:
                model.l1_weights.requires_grad = True
            else:
                model.l1_weights.weight.requires_grad = True
            model.l2_weights_const.requires_grad = False
            model.l2_weights_stim.requires_grad = False
            loss_l1.backward(retain_graph=True)
            if tonotopy:
                w1_grad = model.l1_weights.grad.clone().detach().cpu().numpy().copy()
            else:
                w1_grad = model.l1_weights.weight.grad.clone().detach().cpu().numpy().copy()
            optimizer.step()

            # Calculate the loss for W2_const
            curr_const_q_values = const_q_values.clone().detach().cpu().numpy().copy()
            const_corticostriatal_loss = reward - curr_const_q_values[action]
            target_const_q_values = curr_const_q_values.copy()
            target_const_q_values[action] = target_const_q_values[action] + const_corticostriatal_loss
            target_const_q_values = torch.tensor(target_const_q_values, dtype=torch.float32).to(device)

            loss_l2_const = criterion(const_q_values, target_const_q_values)
            expected_loss_l2_const = 1/3 * (const_corticostriatal_loss) ** 2
            assert np.isclose(loss_l2_const.item(), expected_loss_l2_const, atol=1e-03), f"Expected: {expected_loss_l2_const}, Got: {loss_l2_const.item()}"

            # Update W2_const
            optimizer.zero_grad()
            if tonotopy:
                model.l1_weights.requires_grad = False
            else:
                model.l1_weights.weight.requires_grad = False
            model.l2_weights_const.requires_grad = True
            model.l2_weights_stim.requires_grad = False
            loss_l2_const.backward(retain_graph=True)
            w2_const_grad = model.l2_weights_const.grad.clone().detach().cpu().numpy().copy()
            optimizer.step()

            # Calculate the loss for W2_stim
            curr_stim_q_values = stim_q_values.clone().detach().cpu().numpy().copy()
            stim_corticostriatal_loss = reward - curr_stim_q_values[action]
            target_stim_q_values = curr_stim_q_values.copy()
            target_stim_q_values[action] = target_stim_q_values[action] + stim_corticostriatal_loss
            target_stim_q_values = torch.tensor(target_stim_q_values, dtype=torch.float32).to(device)

            loss_l2_stim = criterion(stim_q_values, target_stim_q_values)
            expected_loss_l2_stim = 1/3 * (stim_corticostriatal_loss) ** 2
            assert np.isclose(loss_l2_stim.item(), expected_loss_l2_stim, atol=1e-03), f"Expected: {expected_loss_l2_stim}, Got: {loss_l2_stim.item()}"

            # Update W2_stim
            optimizer.zero_grad()
            if tonotopy:
                model.l1_weights.requires_grad = False
            else:
                model.l1_weights.weight.requires_grad = False
            model.l2_weights_const.requires_grad = False
            model.l2_weights_stim.requires_grad = True
            loss_l2_stim.backward()
            w2_stim_grad = model.l2_weights_stim.grad.clone().detach().cpu().numpy().copy()
            optimizer.step()

            # Need to unfreeze all the weights again
            if tonotopy:
                model.l1_weights.requires_grad = True
            else:
                model.l1_weights.weight.requires_grad = True
            model.l2_weights_const.requires_grad = True
            model.l2_weights_stim.requires_grad = True

            if tonotopy:
                new_w1 = model.l1_weights.clone().detach().cpu().numpy().copy()
                new_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                new_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()
            else:
                new_w1 = model.l1_weights.weight.clone().detach().cpu().numpy().copy()
                new_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                new_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()

            delta_w1 = new_w1 - old_w1
            delta_w2_const = new_w2_const - old_w2_const
            delta_w2_stim = new_w2_stim - old_w2_stim

            expected_w1_grad = np.zeros(shape=delta_w1.shape)
            expected_w2_const_grad = np.zeros(shape=delta_w2_const.shape)
            expected_w2_stim_grad = np.zeros(shape=delta_w2_stim.shape)

            if tonotopy:
                expected_w2_const_grad[action] = old_w1[0]
                for i in range(num_notes + 1):
                    if i == 0:
                        expected_w1_grad[i] = old_w2[action,0]
                    else:
                        expected_w1_grad[i] = curr_stimulus[i] * old_w2[action,i]
                        expected_w2_stim_grad[action, i - 1] = curr_stimulus[i] * old_w1[i]
            else:
                curr_stimulus_array = curr_stimulus.clone().detach().cpu().numpy().copy()
                expected_w2_const_grad[action] = old_w1[0,0] * curr_stimulus_array[0]
                for i in range(num_notes + 1):
                    if i == 0:
                        expected_w1_grad[:,i] = old_w2[action,:]
                    else:
                        expected_w1_grad[:,i] = curr_stimulus_array[i] * old_w2[action,:]
                        expected_w2_stim_grad[action, i - 1] = old_w1[i,1:] @ curr_stimulus_array[1:]
            
            expected_w1_grad = -1 * expected_w1_grad * 2/3 * cortical_loss
            expected_w2_const_grad = -1 * expected_w2_const_grad * 2/3 * const_corticostriatal_loss
            expected_w2_stim_grad = -1 * expected_w2_stim_grad * 2/3 * stim_corticostriatal_loss
            assert np.allclose(w1_grad, expected_w1_grad, atol=1e-5), f"Expected: {expected_w1_grad}, Got: {w1_grad}"
            assert np.allclose(w2_const_grad, expected_w2_const_grad, atol=1e-5), f"Expected: {expected_w2_const_grad}, Got: {w2_const_grad}"
            assert np.allclose(w2_stim_grad, expected_w2_stim_grad, atol=1e-5), f"Expected: {expected_w2_stim_grad}, Got: {w2_stim_grad}"
            
            expected_delta_w1 = -1 * learning_rate * expected_w1_grad
            expected_delta_w2_const = -1 * learning_rate *  expected_w2_const_grad
            expected_delta_w2_stim = -1 * learning_rate * expected_w2_stim_grad
            assert np.allclose(delta_w1, expected_delta_w1, atol=1e-5), f"Expected: {expected_delta_w1}, Got: {delta_w1}"
            assert np.allclose(delta_w2_const, expected_delta_w2_const, atol=1e-5), f"Expected: {expected_delta_w2_const}, Got: {delta_w2_const}"
            assert np.allclose(delta_w2_stim, expected_delta_w2_stim, atol=1e-5), f"Expected: {expected_delta_w2_stim}, Got: {delta_w2_stim}"

            if spectrogram:
                trial_data = {
                    "model": model,
                    "loss_l1": loss_l1.item(),
                    "loss_l2_const": loss_l2_const.item(),
                    "loss_l2_stim": loss_l2_stim.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "curr_theta": curr_theta,
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward,
                    "reward_volume": reward_volume,
                    "action_penalty": action_penalty
                }
            else:
                trial_data = {
                    "model": model,
                    "loss_l1": loss_l1.item(),
                    "loss_l2_const": loss_l2_const.item(),
                    "loss_l2_stim": loss_l2_stim.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward,
                    "reward_volume": reward_volume,
                    "action_penalty": action_penalty
                }
            data.append(trial_data)
    
    if save_data:
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

def run_deep_rl_experiment(spectrogram=True, task_id=0, thetas=[0,90], model_path=None, num_notes=7, p_train=0.8, p_reward_train=1, p_reward_test_validation=0.5,
                            num_trials=10000, learning_rate=0.1, beta=2.5, rpe_type="full", tonotopy=False, save_data=True, save_path=None):
    """This function runs an experiment similar to that used to train the animals for a deep network trained via reinforcement learning.

    Args:
        spectrogram (bool, optional): If true, then the network trains on the full auditory task; if false, the network trains on the simpler, general task. Defaults to True.
        task_id (int, optional): The number of the task. Either 0 (task 1) or 1 (task 2). Defaults to 0.
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        model_path (pathlib Path object): Path to the previously trained network. Needed if task_id is not zero. Defaults to None.
        num_notes (int, optional): Number of notes used to create the polyphonic sound. Defaults to 7.
        p_train (float, optional): Probability of a train trial (vs. test/validation trial). Defaults to 0.8.
        p_reward_train (float, optional): Probability that the current trial is rewarded for train trials. Defaults to 1.
        p_reward_test_validation (float, optional): Probability that the current trial is rewarded for validation and test trials. Defaults to 0.5.
        num_trials (int, optional): Number of trials in the experiment. Defaults to 10000.
        learning_rate (float, optional): Learning rate for the network. Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter for the softmax action selection. Defaults to 2.5.
        rpe_type (str, optional): Specifies the type of the RPE signal, either "full" or "partial". Defaults to "full".
        tonotopy (bool, optional): If true, then the first layer weights are diagonal, motivated by the existence of tonotopy in auditory cortex. Defaults to False.
        save_data (bool, optional): If true, after every iteration, this function saves a dictionary with relevant trial variables. Defaults to True.
        save_path (pathlib Path object): Path to where data should be saved. Defaults to None.
    """
    # If training on the basic task (input indicates just the presence of the left or right stimulus), then the input dimension
    # (parametrized in the networks as as num_notes) is 2.
    if not spectrogram:
        num_notes = 2

    # If this is the first task, then initialize a new network
    if task_id == 0:
        model = DeepRLAuditoryDiscriminationNetwork(rpe_type=rpe_type, tonotopy=tonotopy, num_notes=num_notes)
    # However, if this is one of the later tasks, load a previously trained network
    else:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        model = data[-1]["model"]
        # Check that the model is the correct type
        assert(type(model) == DeepRLAuditoryDiscriminationNetwork)
        assert(model.rpe_type == rpe_type)
        assert(model.tonotopy == tonotopy)
    
    # Check to see if GPU is available; otherwise, use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Start by keeping track of the initialized model
    if task_id == 0:
        data = [{
            "model": model
        }]
    else:
        data = [{
            "model": model,
            "model_path": model_path
        }]
    
    if model.rpe_type == "full":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif model.rpe_type == "partial":
        optimizer = optim.SGD([
            {'params': [model.l1_weights] if model.tonotopy else model.l1_weights.parameters()},
            {'params': [model.l2_weights_const]},
            {'params': [model.l2_weights_stim]}
        ], lr=learning_rate)
    criterion = nn.MSELoss()
    
    # In the reinforcement version of the model, the output of the network reflects the Q-values associated with choosing left of right,
    # which are updated based on the RPE.
    for i in range(num_trials):
        # Determine the trial type, either train, test, or validation
        trial_type = get_trial_type(p_train=p_train)

        # Choose a stimulus, keeping track of the current theta on the ring and the correct choice given an optimal
        # linear decision boundary
        if spectrogram:
            curr_stimulus, curr_theta, correct_choice = get_auditory_stimulus(trial_type, task_id, thetas)
        else:
            curr_stimulus, correct_choice = get_general_stimulus()
        curr_stimulus = curr_stimulus.to(device)
        
        # The output of the network are the Q-values associated with choosing
        # left or right.
        q_values = model(curr_stimulus)
        curr_q_values = q_values.clone().detach().cpu().numpy().copy()
        
        # Then select an action through a softmax function.
        action, action_probabilities = select_action(curr_q_values, beta=beta)
        
        # Then determine if the choice is rewarded
        if spectrogram:
            reward = get_auditory_reward(trial_type, curr_theta, action, task_id, thetas, p_reward_train=p_reward_train, p_reward_test_validation=p_reward_test_validation)
        else:
            reward = get_general_reward(trial_type, action, correct_choice, p_reward_train=p_reward_train, p_reward_test_validation=p_reward_test_validation)
        
        # If we're using the full RPE, we update all the weights based on the same loss function
        if model.rpe_type == "full":
            # Update the relevant Q-value based on the RPE
            target_q_values = curr_q_values.copy()
            target_q_values[action] = curr_q_values[action] + (reward - curr_q_values[action])
            target_q_values = torch.tensor(target_q_values, dtype=torch.float32).to(device)
            loss = criterion(q_values, target_q_values)
            assert np.isclose(loss.item(), 0.5*(reward - curr_q_values[action])**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if spectrogram:
                trial_data = {
                    "model": model,
                    "loss": loss.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "curr_theta": curr_theta,
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward
                }
            else:
                trial_data = {
                    "model": model,
                    "loss": loss.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward,
                }
            data.append(trial_data)
        
        # If we're using partial RPEs, we update W1, W2_const, and W2_stim independently, using three
        # different loss functions
        elif model.rpe_type == "partial":
            
            if tonotopy:
                old_w1 = model.l1_weights.clone().detach().cpu().numpy().copy()
                old_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                old_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()
                old_w2 = np.concatenate((old_w2_const, old_w2_stim), axis=1)
            else:
                old_w1 = model.l1_weights.weight.clone().detach().cpu().numpy().copy()
                old_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                old_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()
                old_w2 = np.concatenate((old_w2_const, old_w2_stim), axis=1)

            # Compute the Q-values using only the const term
            const_term_input = curr_stimulus.clone()
            const_term_input[1:] = 0  # Zero out stimulus terms
            const_term_input.to(device)
            const_q_values = model(const_term_input)
            if tonotopy:
                expected_const_q_values = old_w1[0] * old_w2[:,0]
            else:
                expected_const_q_values = old_w2 @ old_w1[:,0]
            assert np.allclose(const_q_values.clone().detach().cpu().numpy().copy(), expected_const_q_values, atol=1e-03), f"Expected: {expected_const_q_values}, Got: {const_q_values.clone().detach().cpu().numpy().copy()}"

            # Compute the Q-values using only the stim term
            stim_term_input = curr_stimulus.clone()
            stim_term_input[0] = 0  # Zero out the constant term
            stim_term_input.to(device)
            stim_q_values = model(stim_term_input)
            expected_stim_q_values = np.zeros(shape=stim_q_values.clone().detach().cpu().numpy().copy().shape)
            for i in range(1, num_notes + 1):
                if tonotopy:
                    expected_stim_q_values = expected_stim_q_values + curr_stimulus[i].clone().detach().cpu().numpy().copy() * old_w1[i] * old_w2[:,i]
                else:
                    expected_stim_q_values = expected_stim_q_values + curr_stimulus[i].clone().detach().cpu().numpy().copy() * old_w2 @ old_w1[:,i]
            assert np.allclose(stim_q_values.clone().detach().cpu().numpy().copy(), expected_stim_q_values, atol=1e-03), f"Expected: {expected_stim_q_values}, Got: {stim_q_values.clone().detach().cpu().numpy().copy()}"

            # Calculate the loss for W1
            cortical_loss = reward - curr_q_values[action]
            target_cortical_q_values = curr_q_values.copy()
            target_cortical_q_values[action] = curr_q_values[action] + cortical_loss
            target_cortical_q_values = torch.tensor(target_cortical_q_values, dtype=torch.float32).to(device)

            loss_l1 = criterion(q_values, target_cortical_q_values)
            expected_loss_l1 = 0.5 * (cortical_loss) ** 2
            assert np.isclose(loss_l1.item(), expected_loss_l1, atol=1e-03), f"Expected: {expected_loss_l1}, Got: {loss_l1.item()}"

            # Update W1
            optimizer.zero_grad()
            if tonotopy:
                model.l1_weights.requires_grad = True
            else:
                model.l1_weights.weight.requires_grad = True
            model.l2_weights_const.requires_grad = False
            model.l2_weights_stim.requires_grad = False
            loss_l1.backward(retain_graph=True)
            if tonotopy:
                w1_grad = model.l1_weights.grad.clone().detach().cpu().numpy().copy()
            else:
                w1_grad = model.l1_weights.weight.grad.clone().detach().cpu().numpy().copy()
            optimizer.step()

            # Calculate the loss for W2_const
            curr_const_q_values = const_q_values.clone().detach().cpu().numpy().copy()
            const_corticostriatal_loss = reward - curr_const_q_values[action]
            target_const_q_values = curr_const_q_values.copy()
            target_const_q_values[action] = target_const_q_values[action] + const_corticostriatal_loss
            target_const_q_values = torch.tensor(target_const_q_values, dtype=torch.float32).to(device)

            loss_l2_const = criterion(const_q_values, target_const_q_values)
            expected_loss_l2_const = 0.5 * (const_corticostriatal_loss) ** 2
            assert np.isclose(loss_l2_const.item(), expected_loss_l2_const, atol=1e-03), f"Expected: {expected_loss_l2_const}, Got: {loss_l2_const.item()}"

            # Update W2_const
            optimizer.zero_grad()
            if tonotopy:
                model.l1_weights.requires_grad = False
            else:
                model.l1_weights.weight.requires_grad = False
            model.l2_weights_const.requires_grad = True
            model.l2_weights_stim.requires_grad = False
            loss_l2_const.backward(retain_graph=True)
            w2_const_grad = model.l2_weights_const.grad.clone().detach().cpu().numpy().copy()
            optimizer.step()

            # Calculate the loss for W2_stim
            curr_stim_q_values = stim_q_values.clone().detach().cpu().numpy().copy()
            stim_corticostriatal_loss = reward - curr_stim_q_values[action]
            target_stim_q_values = curr_stim_q_values.copy()
            target_stim_q_values[action] = target_stim_q_values[action] + stim_corticostriatal_loss
            target_stim_q_values = torch.tensor(target_stim_q_values, dtype=torch.float32).to(device)

            loss_l2_stim = criterion(stim_q_values, target_stim_q_values)
            expected_loss_l2_stim = 0.5 * (stim_corticostriatal_loss) ** 2
            assert np.isclose(loss_l2_stim.item(), expected_loss_l2_stim, atol=1e-03), f"Expected: {expected_loss_l2_stim}, Got: {loss_l2_stim.item()}"

            # Update W2_stim
            optimizer.zero_grad()
            if tonotopy:
                model.l1_weights.requires_grad = False
            else:
                model.l1_weights.weight.requires_grad = False
            model.l2_weights_const.requires_grad = False
            model.l2_weights_stim.requires_grad = True
            loss_l2_stim.backward()
            w2_stim_grad = model.l2_weights_stim.grad.clone().detach().cpu().numpy().copy()
            optimizer.step()

            # Need to unfreeze all the weights again
            if tonotopy:
                model.l1_weights.requires_grad = True
            else:
                model.l1_weights.weight.requires_grad = True
            model.l2_weights_const.requires_grad = True
            model.l2_weights_stim.requires_grad = True

            if tonotopy:
                new_w1 = model.l1_weights.clone().detach().cpu().numpy().copy()
                new_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                new_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()
            else:
                new_w1 = model.l1_weights.weight.clone().detach().cpu().numpy().copy()
                new_w2_const = model.l2_weights_const.clone().detach().cpu().numpy().copy()
                new_w2_stim = model.l2_weights_stim.clone().detach().cpu().numpy().copy()

            delta_w1 = new_w1 - old_w1
            delta_w2_const = new_w2_const - old_w2_const
            delta_w2_stim = new_w2_stim - old_w2_stim

            expected_w1_grad = np.zeros(shape=delta_w1.shape)
            expected_w2_const_grad = np.zeros(shape=delta_w2_const.shape)
            expected_w2_stim_grad = np.zeros(shape=delta_w2_stim.shape)

            if tonotopy:
                expected_w2_const_grad[action] = old_w1[0]
                for i in range(num_notes + 1):
                    if i == 0:
                        expected_w1_grad[i] = old_w2[action,0]
                    else:
                        expected_w1_grad[i] = curr_stimulus[i] * old_w2[action,i]
                        expected_w2_stim_grad[action, i - 1] = curr_stimulus[i] * old_w1[i]
            else:
                curr_stimulus_array = curr_stimulus.clone().detach().cpu().numpy().copy()
                expected_w2_const_grad[action] = old_w1[0,0] * curr_stimulus_array[0]
                for i in range(num_notes + 1):
                    if i == 0:
                        expected_w1_grad[:,i] = old_w2[action,:]
                    else:
                        expected_w1_grad[:,i] = curr_stimulus_array[i] * old_w2[action,:]
                        expected_w2_stim_grad[action, i - 1] = old_w1[i,1:] @ curr_stimulus_array[1:]
            
            expected_w1_grad = -1 * expected_w1_grad * cortical_loss
            expected_w2_const_grad = -1 * expected_w2_const_grad * const_corticostriatal_loss
            expected_w2_stim_grad = -1 * expected_w2_stim_grad * stim_corticostriatal_loss
            assert np.allclose(w1_grad, expected_w1_grad, atol=1e-5), f"Expected: {expected_w1_grad}, Got: {w1_grad}"
            assert np.allclose(w2_const_grad, expected_w2_const_grad, atol=1e-5), f"Expected: {expected_w2_const_grad}, Got: {w2_const_grad}"
            assert np.allclose(w2_stim_grad, expected_w2_stim_grad, atol=1e-5), f"Expected: {expected_w2_stim_grad}, Got: {w2_stim_grad}"
            
            expected_delta_w1 = -1 * learning_rate * expected_w1_grad
            expected_delta_w2_const = -1 * learning_rate *  expected_w2_const_grad
            expected_delta_w2_stim = -1 * learning_rate * expected_w2_stim_grad
            assert np.allclose(delta_w1, expected_delta_w1, atol=1e-5), f"Expected: {expected_delta_w1}, Got: {delta_w1}"
            assert np.allclose(delta_w2_const, expected_delta_w2_const, atol=1e-5), f"Expected: {expected_delta_w2_const}, Got: {delta_w2_const}"
            assert np.allclose(delta_w2_stim, expected_delta_w2_stim, atol=1e-5), f"Expected: {expected_delta_w2_stim}, Got: {delta_w2_stim}"

            if spectrogram:
                trial_data = {
                    "model": model,
                    "loss_l1": loss_l1.item(),
                    "loss_l2_const": loss_l2_const.item(),
                    "loss_l2_stim": loss_l2_stim.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "curr_theta": curr_theta,
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward
                }
            else:
                trial_data = {
                    "model": model,
                    "loss_l1": loss_l1.item(),
                    "loss_l2_const": loss_l2_const.item(),
                    "loss_l2_stim": loss_l2_stim.item(),
                    "trial_type": trial_type,
                    "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                    "correct_choice": correct_choice,
                    "q_values": curr_q_values,
                    "action": action,
                    "action_probabilities": action_probabilities,
                    "beta": beta,
                    "reward": reward
                }
            data.append(trial_data)
    
    if save_data:
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

def run_shallow_supervised_experiment(spectrogram=True, task_id=0, thetas=[0,90], model_path=None, num_notes=7, p_train=0.8, 
                                        num_trials=10000, learning_rate=0.1, beta=2.5, save_data=True, save_path=None):
    """This function runs an experiment similar to that used to train the animals for a shallow network trained via supervised learning.

    Args:
        spectrogram (bool, optional): If true, then the network trains on the full auditory task; if false, the network trains on the simpler, general task. Defaults to True.
        task_id (int, optional): The number of the task. Either 0 (task 1) or 1 (task 2). Defaults to 0.
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        model_path (pathlib Path object): Path to the previously trained network. Needed if task_id is not zero. Defaults to None.
        num_notes (int, optional): Number of notes used to create the polyphonic sound. Defaults to 7.
        p_train (float, optional): Probability of a train trial (vs. test/validation trial). Defaults to 0.8.
        num_trials (int, optional): Number of trials in the experiment. Defaults to 10000.
        learning_rate (float, optional): Learning rate for the network. Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter for the softmax action selection. Defaults to 2.5.
        save_data (bool, optional): If true, after every iteration, this function saves a dictionary with relevant trial variables. Defaults to True.
        save_path (pathlib Path object): Path to where data should be saved. Defaults to None.
    """
    # If training on the basic task (input indicates just the presence of the left or right stimulus), then the input dimension
    # (parametrized in the networks as as num_notes) is 2.
    if not spectrogram:
        num_notes = 2
    
    # If this is the first task, then initialize a new network
    if task_id == 0:
        model = ShallowSupervisedAuditoryDiscriminationNetwork(num_notes=num_notes)
    # However, if this is one of the later tasks, load a previously trained network
    else:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        model = data[-1]["model"]
        # Check that the model is the correct type
        assert(type(model) == ShallowSupervisedAuditoryDiscriminationNetwork)
    
    # Check to see if GPU is available; otherwise, use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Start by keeping track of the initialized model
    if task_id == 0:
        data = [{
            "model": model
        }]
    else:
        data = [{
            "model": model,
            "model_path": model_path
        }]

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # In the supervised version of the model, the output of the network reflects the probabilities of choosing left or right,
    # which is then directly compared to the correct probabilities, either [1,0] for the left choice or [0,1] for the right choice.
    for i in range(num_trials):
        # Determine the trial type, either train, test, or validation
        trial_type = get_trial_type(p_train=p_train)

        # Choose a stimulus, keeping track of the current theta on the ring and the correct choice given an optimal
        # linear decision boundary
        if spectrogram:
            curr_stimulus, curr_theta, correct_choice = get_auditory_stimulus(trial_type, task_id, thetas)
        else:
            curr_stimulus, correct_choice = get_general_stimulus()
        curr_stimulus = curr_stimulus.to(device)

        # Assign the correct probabilities for each action
        if correct_choice == 0:
            target_action_probabilities = torch.tensor([1,0], dtype=torch.float32).to(device)
        elif correct_choice == 1:
            target_action_probabilities = torch.tensor([0,1], dtype=torch.float32).to(device)
        
        # The output of the network are the probabilities of choosing left or right.
        action_probabilities = model(curr_stimulus)

        # To track the accuracy of the model, select an action based on those probabilities
        action, _ = select_action(action_probabilities.clone().detach().cpu().numpy().copy(), beta=beta)
        
        loss = criterion(action_probabilities, target_action_probabilities)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if spectrogram:
            trial_data = {
                "model": model,
                "loss": loss.item(),
                "trial_type": trial_type,
                "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                "curr_theta": curr_theta,
                "correct_choice": correct_choice,
                "action": action,
                "action_probabilities": action_probabilities.clone().detach().cpu().numpy().copy(),
                "beta": beta
            }
        else:
            trial_data = {
                "model": model,
                "loss": loss.item(),
                "trial_type": trial_type,
                "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                "correct_choice": correct_choice,
                "action": action,
                "action_probabilities": action_probabilities.clone().detach().cpu().numpy().copy(),
                "beta": beta
            }
        data.append(trial_data)
    
    if save_data:
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

def run_deep_supervised_experiment(spectrogram=True, task_id=0, thetas=[0,90], model_path=None, num_notes=7, p_train=0.8, 
                                    num_trials=10000, learning_rate=0.1, beta=2.5, tonotopy=False, save_data=True, save_path=None):
    """This function runs an experiment similar to that used to train the animals for a deep network trained via supervised learning.

    Args:
        spectrogram (bool, optional): If true, then the network trains on the full auditory task; if false, the network trains on the simpler, general task. Defaults to True.
        task_id (int, optional): The number of the task. Either 0 (task 1) or 1 (task 2). Defaults to 0.
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        model_path (pathlib Path object): Path to the previously trained network. Needed if task_id is not zero. Defaults to None.
        num_notes (int, optional): Number of notes used to create the polyphonic sound. Defaults to 7.
        p_train (float, optional): Probability of a train trial (vs. test/validation trial). Defaults to 0.8.
        num_trials (int, optional): Number of trials in the experiment. Defaults to 10000.
        learning_rate (float, optional): Learning rate for the network. Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter for the softmax action selection. Defaults to 2.5.
        tonotopy (bool, optional): If true, then the first layer weights are diagonal, motivated by the existence of tonotopy in auditory cortex. Defaults to False.
        save_data (bool, optional): If true, after every iteration, this function saves a dictionary with relevant trial variables. Defaults to True.
        save_path (pathlib Path object): Path to where data should be saved. Defaults to None.
    """
    # If training on the basic task (input indicates just the presence of the left or right stimulus), then the input dimension
    # (parametrized in the networks as as num_notes) is 2.
    if not spectrogram:
        num_notes = 2
    
    # If this is the first task, then initialize a new network
    if task_id == 0:
        model = DeepSupervisedAuditoryDiscriminationNetwork(tonotopy=tonotopy, num_notes=num_notes)
    # However, if this is one of the later tasks, load a previously trained network
    else:
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        model = data[-1]["model"]
        # Check that the model is the correct type
        assert(type(model) == DeepSupervisedAuditoryDiscriminationNetwork)
        assert(model.tonotopy == tonotopy)
    
    # Check to see if GPU is available; otherwise, use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Start by keeping track of the initialized model
    if task_id == 0:
        data = [{
            "model": model
        }]
    else:
        data = [{
            "model": model,
            "model_path": model_path
        }]

    if tonotopy:
        optimizer = optim.SGD([
                {'params': [model.l1_weights]},
                {'params': model.l2_weights.parameters()},
            ], lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # In the supervised version of the model, the output of the network reflects the probabilities of choosing left or right,
    # which is then directly compared to the correct probabilities, either [1,0] for the left choice or [0,1] for the right choice.
    for i in range(num_trials):
        # Determine the trial type, either train, test, or validation
        trial_type = get_trial_type(p_train=p_train)

        # Choose a stimulus, keeping track of the current theta on the ring and the correct choice given an optimal
        # linear decision boundary
        if spectrogram:
            curr_stimulus, curr_theta, correct_choice = get_auditory_stimulus(trial_type, task_id, thetas)
        else:
            curr_stimulus, correct_choice = get_general_stimulus()
        curr_stimulus = curr_stimulus.to(device)

        # Assign the correct probabilities for each action
        if correct_choice == 0:
            target_action_probabilities = torch.tensor([1,0], dtype=torch.float32).to(device)
        elif correct_choice == 1:
            target_action_probabilities = torch.tensor([0,1], dtype=torch.float32).to(device)
        
        # The output of the network are the probabilities of choosing left or right.
        action_probabilities = model(curr_stimulus)

        # To track the accuracy of the model, select an action based on those probabilities
        action, _ = select_action(action_probabilities.clone().detach().cpu().numpy().copy(), beta=beta)
        
        loss = criterion(action_probabilities, target_action_probabilities)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if spectrogram:
            trial_data = {
                "model": model,
                "loss": loss.item(),
                "trial_type": trial_type,
                "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                "curr_theta": curr_theta,
                "correct_choice": correct_choice,
                "action": action,
                "action_probabilities": action_probabilities.clone().detach().cpu().numpy().copy(),
                "beta": beta
            }
        else:
            trial_data = {
                "model": model,
                "loss": loss.item(),
                "trial_type": trial_type,
                "curr_stimulus": curr_stimulus.clone().detach().cpu().numpy().copy(),
                "correct_choice": correct_choice,
                "action": action,
                "action_probabilities": action_probabilities.clone().detach().cpu().numpy().copy(),
                "beta": beta
            }
        data.append(trial_data)
    
    if save_data:
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

def run_experiment(spectrogram=True, task_id=0, thetas=[0,90], model_path=None, num_notes=7, p_train=0.8, p_reward_train=1, p_reward_test_validation=0.5,
                    num_trials=10000, learning_rate=0.1, beta=2.5, depth=True, rpe=True, rpe_type="full", with_inaction=False, reward_volume=20,
                    action_penalty=5, tonotopy=False, save_data=True, save_path=None):
    """This function runs an experiment similar to that used to train the animals.

    Args:
        spectrogram (bool, optional): If true, then the network trains on the full auditory task; if false, the network trains on the simpler, general task. Defaults to True.
        task_id (int, optional): The number of the task. Either 0 (task 1) or 1 (task 2). Defaults to 0.
        thetas (list, optional): Angles for the right choice sounds for tasks 1 (element 0) and 2 (element 1). 
                                 The left choice sounds are just those angles + 180 degrees. Defaults to [0,90].
        model_path (pathlib Path object): Path to the previously trained network. Needed if task_id is not zero. Defaults to None.
        num_notes (int, optional): Number of notes used to create the polyphonic sound. Defaults to 7.
        p_train (float, optional): Probability of a train trial (vs. test/validation trial). Defaults to 0.8.
        p_reward_train (float, optional): Probability that the current trial is rewarded for train trials. Defaults to 1.
        p_reward_test_validation (float, optional): Probability that the current trial is rewarded for validation and test trials. Defaults to 0.5.
        num_trials (int, optional): Number of trials in the experiment. Defaults to 10000.
        learning_rate (float, optional): Learning rate for the network. Defaults to 0.1.
        beta (float, optional): Inverse temperature parameter for the softmax action selection. Defaults to 2.5.
        depth (bool, optional): If true, the network trained is a deep network; if false, the network trained is a shallow network. Defaults to True.
        rpe (bool, optional): If true, then the network learns via reinforcement learning; if false, the network learns via supervised learning. Defaults to True.
        rpe_type (str, optional): Specifies the type of the RPE signal, either "full" or "partial". Defaults to "full".
        with_inaction (bool, optional): If true, then the network has three outputs, representing the Q-values for a left choice, a right choice, and inaction. Defaults to False.
        reward_volume (int, optional): Only relevant if with_inaction is True. Amount of reward delivered. Defaults to 20.
        action_penalty (int, optional): Only relevant if with_inaction is True. Cost of acting. Defaults to 5.
        tonotopy (bool, optional): If true, then the first layer weights are diagonal, motivated by the existence of tonotopy in auditory cortex. Defaults to False.
        save_data (bool, optional): If true, after every iteration, this function saves a dictionary with relevant trial variables. Defaults to True.
        save_path (pathlib Path object): Path to where data should be saved. Defaults to None.
    """
    if rpe:
        if depth:
            if with_inaction:
                run_deep_rl_with_inaction_experiment(spectrogram=spectrogram, task_id=task_id, thetas=thetas, model_path=model_path, num_notes=num_notes, 
                                                      p_train=p_train, p_reward_train=p_reward_train, p_reward_test_validation=p_reward_test_validation,
                                                      num_trials=num_trials, learning_rate=learning_rate, beta=beta, reward_volume=reward_volume,
                                                      action_penalty=action_penalty, rpe_type=rpe_type, tonotopy=tonotopy, save_data=save_data, save_path=save_path)
            else:
                run_deep_rl_experiment(spectrogram=spectrogram, task_id=task_id, thetas=thetas, model_path=model_path, num_notes=num_notes, 
                                        p_train=p_train, p_reward_train=p_reward_train, p_reward_test_validation=p_reward_test_validation,
                                        num_trials=num_trials, learning_rate=learning_rate, rpe_type=rpe_type, 
                                        tonotopy=tonotopy, save_data=save_data, save_path=save_path)
        else:
            run_shallow_rl_experiment(spectrogram=spectrogram, task_id=task_id, thetas=thetas, model_path=model_path, num_notes=num_notes, 
                                       p_train=p_train, num_trials=num_trials, learning_rate=learning_rate, rpe_type=rpe_type, 
                                       save_data=save_data, save_path=save_path)
    else:
        if depth:
            run_deep_supervised_experiment(spectrogram=spectrogram, task_id=task_id, thetas=thetas, model_path=model_path, num_notes=num_notes, 
                                            p_train=p_train, num_trials=num_trials, learning_rate=learning_rate, 
                                            tonotopy=tonotopy, save_data=save_data, save_path=save_path)
        else:
            run_shallow_supervised_experiment(spectrogram=spectrogram, task_id=task_id, thetas=thetas, model_path=model_path, num_notes=num_notes, 
                                               p_train=p_train, num_trials=num_trials, learning_rate=learning_rate,
                                               save_data=save_data, save_path=save_path)