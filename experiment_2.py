from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

# environment = Environment.from_conda_specification(name="sentiment-env", file_path="conda.yml")
# environment.register(ws)
environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="experiment", 
    entry_script="experiment.py", 
    compute_target="local", 
    framework_version="2.1",  
    script_params={'--n-words': 5000, '--epochs': 2},
    environment_definition=environment
    )

experiment = Experiment(workspace=ws, name="sentiment-analysis")
run = experiment.submit(config=estimator)

run.wait_for_completion(show_output=True)