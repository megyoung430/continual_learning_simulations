from pathlib import Path
import src.simulations.experiment_fxns as exp

FILE_NAME = "Full RPE Model.pk1"
save_path = Path("/Users/megyoung/continual_learning_simulations/results/trained_models/" + FILE_NAME)
exp.run_experiment(task_id=0, thetas=[0,90], spectrogram=False, num_notes=7, p_train=1, num_trials=10000, learning_rate=0.05, rpe=True, rpe_type="full", tonotopy=True, save_data=True, save_path=save_path)
