import pickle
from pathlib import Path
import src.visualization.plotting_fxns as plot
import src.simulations.analysis_fxns as anal

data_directory = Path("/Users/megyoung/continual_learning_simulations/results/trained_models/reward_structure/")
model_names = ["p_reward_train=0.9 and p_reward_test_validation=0.5",
          "p_reward_train=0.9 and p_reward_test_validation=0.0",
          "p_reward_train=1.0 and p_reward_test_validation=0.0"]
data_paths_across_models = {}

for model_name in model_names:

    curr_data_paths = anal.get_all_models(data_directory, model_name)
    data_paths_across_models[model_name] = curr_data_paths
    
plot.plot_validation_accuracy_comparison_across_models(model_names, data_paths_across_models, with_window=True)

"""
def check_p_reward_train(data_path):
    with open(data_path, 'rb') as file:
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
    print(num_rewarded_correct_train_trials/num_correct_train_trials)

for path in data_paths_across_models[model_names[0]]:
    check_p_reward_train(path)
"""