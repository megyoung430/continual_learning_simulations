from pathlib import Path
import src.models

FILE_NAME = "Supervised Model ikjkj.pk1"
save_path = Path("/Users/megyoung/continual-learning-simulations/results/" + FILE_NAME)
src.models.run_experiment(task_id=0, thetas=[0,90], num_notes=7, p_train=1, num_trials=10000, learning_rate=0.1, rpe=False, rpe_type="partial", tonotopy=True, save_data=True, save_path=save_path) 