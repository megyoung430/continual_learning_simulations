"""This file allows you to simulate different reward structures during training on Task 1.

The three discussed in the meeting:
1) Rewarding correct training trials with 0.9 probability and rewarding all test and validation trials with 0.5 probability.
2) Rewarding correct training trials with 0.9 probability and never rewarding any test or validation trial.
3) Always rewarding correct training trials and never rewarding any test or validation trial.
"""

from pathlib import Path
import src.simulations.experiment_fxns as exp

num_simulations = 5
spectrogram = True
task_id = 0
thetas = [0,90]
num_notes = 7
p_train = 0.8
p_reward_train = 1
p_reward_test_validation = 0.0
num_trials = 10000
learning_rate = 0.05
beta = 2.5
depth = True
rpe = True
rpe_type = "partial"
tonotopy = True

base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/reward_structure/"

for i in range(num_simulations):
    print("Training Network " + str(i))
    FILE_NAME = "Task " + str(task_id) + ", Theta " + str(thetas[task_id]) + " with p_reward_train= " + str(p_reward_train) + " and p_reward_test_validation=" + str(p_reward_test_validation) + " (" + str(i) + ")" + ".pk1"
    save_path = Path(base_path + FILE_NAME)
    exp.run_experiment(spectrogram=spectrogram, task_id=task_id, thetas=thetas, num_notes=num_notes, p_train=p_train, 
                        p_reward_train=p_reward_train, p_reward_test_validation=p_reward_test_validation, num_trials=num_trials, 
                        learning_rate=learning_rate, beta=beta, depth=depth, rpe=rpe, rpe_type=rpe_type, tonotopy=tonotopy, save_data=True, save_path=save_path)