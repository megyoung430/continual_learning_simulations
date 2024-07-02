from pathlib import Path
import src.visualization.plotting_fxns as plot

FILE_NAME = "Full RPE Model.pk1"
data_path = Path("/Users/megyoung/continual_learning_simulations/results/trained_models/" + FILE_NAME)
plot.plot_summary_figure(data_path)