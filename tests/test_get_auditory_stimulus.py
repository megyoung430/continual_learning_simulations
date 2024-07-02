import numpy as np
import pytest
from src.simulations import experiment_fxns as exp

def test_get_auditory_stimulus():
    
    trial_type = "train"
    theta = 0
    num_notes = 7
    optimal_order = np.asarray([1, 4, 7, 2, 3, 5, 6]) - 1
    
    # Explicitly calculate the correct auditory stimulus
    positions = np.linspace(0., 360., num_notes + 1)[:-1]
    positions = np.deg2rad(positions) + np.deg2rad(theta)
    stim = (0.5 * (np.cos(positions) + 1.))[optimal_order]
    expected_stim = np.ones(shape=(num_notes + 1))
    expected_stim[1:] = stim

    # Generate the actual auditory stimulus
    actual_stim, _ , _ = exp.get_auditory_stimulus(trial_type, task_id=0, thetas=[0,90], p_stim_right=1, num_notes=num_notes)
    actual_stim = actual_stim.clone().detach().numpy().copy()
    assert pytest.approx(actual_stim) == expected_stim