from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.train.hyperdrive import GridParameterSampling, choice, MedianStoppingPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

# environment = Environment.from_conda_specification(name="sentiment-env", file_path="experiment-env.yml")
# environment.register(ws)
environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="imdb",
    entry_script="experiment.py",
    compute_target="archi-trainer",
    framework_version="2.1",
    environment_definition=environment
)

param_space = {
    '--epochs': choice(2, 3, 5,10),
    '--n-words': choice(5000, 20000, 50000, 80000),
    '--dim-embedding':choice(32, 64)
}

param_sampling = GridParameterSampling(param_space)
early_termination_policy = MedianStoppingPolicy(
    evaluation_interval=1, delay_evaluation=5)

hyperdrive = HyperDriveConfig(
    estimator=estimator,
    primary_metric_name='accuracy', 
    hyperparameter_sampling=param_sampling,
    policy=early_termination_policy, 
    max_total_runs=12, 
    max_concurrent_runs=4,
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE)


experiment = Experiment(workspace=ws, name="sentiment-analysis")
run = experiment.submit(config=hyperdrive)

run.wait_for_completion(show_output=True)

for child_run in run.get_children():
    print(child_run.id, child_run.get_metrics())

best_run = run.get_best_run_by_primary_metric()