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

plot.plot_train_accuracy_comparison_across_models(model_names, data_paths_across_models, with_window=False)   
plot.plot_train_accuracy_comparison_across_models(model_names, data_paths_across_models, with_window=True)
plot.plot_validation_accuracy_comparison_across_models(model_names, data_paths_across_models, with_window=False)
plot.plot_validation_accuracy_comparison_across_models(model_names, data_paths_across_models, with_window=True)