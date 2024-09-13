from pathlib import Path
import pickle
import src.simulations.forgetting_fxns as forg

num_simulations = 5
spectrogram = True
depth = False
rpe = True
rpe_type = "partial"
tonotopy = False
#base_path = "/ceph/saxe/myoung/continual_learning_simulations/"
base_path = "/Users/megyoung/continual_learning_simulations/"

if depth:
    if rpe:
        if rpe_type == "full":
            if spectrogram:
                if tonotopy:
                    base_path = base_path + "results/trained_models/deep_rl/full_rpe/full_task/diagonal/"
                else:
                    base_path = base_path + "results/trained_models/deep_rl/full_rpe/full_task/fully_connected/"
            else:
                base_path = base_path + "results/trained_models/deep_rl/full_rpe/simplified_task/"
        else:
            if spectrogram:
                if tonotopy:
                    base_path = base_path + "results/trained_models/deep_rl/partial_rpe/full_task/diagonal/"
                else:
                    base_path = base_path + "results/trained_models/deep_rl/partial_rpe/full_task/fully_connected/"
            else:
                base_path = base_path + "results/trained_models/deep_rl/partial_rpe/simplified_task/"
    else:
        if spectrogram:
            if tonotopy:
                base_path = base_path + "results/trained_models/deep_supervised/full_task/diagonal/"
            else:
                base_path = base_path + "results/trained_models/deep_supervised/full_task/fully_connected/"
        else:
            base_path = base_path + "results/trained_models/deep_supervised/simplified_task/"
else:
    if rpe:
        if rpe_type == "full":
            if spectrogram:
                base_path = base_path + "results/trained_models/shallow_rl/full_rpe/full_task/"
            else:
                base_path = base_path + "results/trained_models/shallow_rl/full_rpe/simplified_task/"
        else:
            if spectrogram:
                base_path = base_path + "results/trained_models/shallow_rl/partial_rpe/full_task/"
            else:
                base_path = base_path + "results/trained_models/shallow_rl/partial_rpe/simplified_task/"
    else:
        if spectrogram:
            base_path = base_path + "results/trained_models/shallow_supervised/full_task/"
        else:
            base_path = base_path + "results/trained_models/shallow_supervised/simplified_task/"

all_thetas = [[0, i] for i in range(15, 361, 15)]

for i in range(num_simulations):
    for j in range(num_simulations):
        for theta in all_thetas:
            # Define the paths
            MODEL_FILE_NAME = "Task 1, Theta " + str(theta[1]) + " from Task 0, Theta 0 (" + str(i) + ") (" + str(j) + ").pk1"
            model_path = Path(base_path + MODEL_FILE_NAME)
            RESULTS_FILE_NAME = "Forgetting Results for " + MODEL_FILE_NAME
            save_path = Path(base_path + RESULTS_FILE_NAME)

            # Assess forgetting for that particular model
            if save_path.exists():
                print("Forgetting results for " + MODEL_FILE_NAME + " already exists.")
            else:
                print("Analyzing forgetting for " + MODEL_FILE_NAME)
                actions_across_points, curr_thetas_across_points, correct_choices_across_points, performances_across_points = forg.assess_forgetting(model_path)
            
                # Store the results in a dictionary
                forgetting_results = {}
                forgetting_results["actions"] = actions_across_points
                forgetting_results["curr_thetas"] = curr_thetas_across_points
                forgetting_results["correct_choices"] = correct_choices_across_points
                forgetting_results["performances"] = performances_across_points
                
                # Save the results
                with open(save_path, 'wb') as pickle_file:
                    pickle.dump(forgetting_results, pickle_file)