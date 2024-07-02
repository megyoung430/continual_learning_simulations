import numpy as np
from src.simulations import experiment_fxns as exp

def test_select_action_random():
    q_values = np.ones(shape=(2))
    _, probabilities = exp.select_action(q_values, beta=0)
    expected_probabilities = 0.5*np.ones(shape=(2))
    assert np.allclose(probabilities, expected_probabilities)