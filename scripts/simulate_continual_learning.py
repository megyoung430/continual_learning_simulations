from pathlib import Path
import src.simulations.experiment_fxns as exp

num_simulations = 5
spectrogram = True
task_id = 1
thetas = [0,90]
num_notes = 7
p_train = 1
num_trials = 10000
learning_rate = 0.05
beta = 2.5
depth = True
rpe = True
rpe_type = "partial"
tonotopy = True

if depth:
    if rpe:
        if rpe_type == "full":
            if spectrogram:
                if tonotopy:
                    base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_rl/full_rpe/full_task/diagonal/"
                else:
                    base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_rl/full_rpe/full_task/fully_connected/"
            else:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_rl/full_rpe/simplified_task/"
        else:
            if spectrogram:
                if tonotopy:
                    base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_rl/partial_rpe/full_task/diagonal/"
                else:
                    base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_rl/partial_rpe/full_task/fully_connected/"
            else:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_rl/partial_rpe/simplified_task/"
    else:
        if spectrogram:
            if tonotopy:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_supervised/full_task/diagonal/"
            else:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_supervised/full_task/fully_connected/"
        else:
            base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/deep_supervised/simplified_task/"
else:
    if rpe:
        if rpe_type == "full":
            if spectrogram:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/shallow_rl/full_rpe/full_task/"
            else:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/shallow_rl/full_rpe/simplified_task/"
        else:
            if spectrogram:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/shallow_rl/partial_rpe/full_task/"
            else:
                base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/shallow_rl/partial_rpe/simplified_task/"
    else:
        if spectrogram:
            base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/shallow_supervised/full_task/"
        else:
            base_path = "/Users/megyoung/continual_learning_simulations/results/trained_models/shallow_supervised/simplified_task/"

ver = 2
old_task_id = 0
MODEL_FILE_NAME = "Task " + str(old_task_id) + ", Theta " + str(thetas[old_task_id]) + " (" + str(ver) + ")" + ".pk1"
model_path = base_path + MODEL_FILE_NAME

for i in range(num_simulations):
    FILE_NAME = "Task " + str(task_id) + ", Theta " + str(thetas[task_id]) + " from Task 0, Theta " + str(thetas[old_task_id]) + " (" + str(ver) + ") (" + str(i) + ")" + ".pk1"
    save_path = Path(base_path + FILE_NAME)
    if save_path.exists():
        print("Network " + str(i) + " already exists.")
    else:
        print("Training Network " + str(i))
        exp.run_experiment(spectrogram=spectrogram, task_id=task_id, thetas=thetas, model_path=model_path, num_notes=num_notes, p_train=p_train, num_trials=num_trials, 
                        learning_rate=learning_rate, beta=beta, depth=depth, rpe=rpe, rpe_type=rpe_type, tonotopy=tonotopy, save_data=True, save_path=save_path)