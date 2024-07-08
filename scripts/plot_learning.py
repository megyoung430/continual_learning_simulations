from pathlib import Path
import src.visualization.plotting_fxns as plot

task_id = 0
thetas = [0,90]
p_reward_train = 0.9
p_reward_test_validation = 0.5
ver = 1

for ver in range(0,5):
    DATA_FILE_NAME = "Task " + str(task_id) + ", Theta " + str(thetas[task_id]) + " with p_reward_train= " + str(p_reward_train) + " and p_reward_test_validation=" + str(p_reward_test_validation) + " (" + str(ver) + ")" + ".pk1"
    data_path = Path("/Users/megyoung/continual_learning_simulations/results/trained_models/reward_structure/" + DATA_FILE_NAME)

    SAVE_FILE_NAME = "Performance.pdf"
    save_path = Path("/Users/megyoung/continual_learning_simulations/results/trained_models/reward_structure/" + SAVE_FILE_NAME)
    plot.plot_summary_figure(data_path, save_path)