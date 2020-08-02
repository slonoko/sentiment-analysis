from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()
fra_eng_ds = ws.datasets['fra-eng-translation']

environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="translator",
    entry_script="experiment.py",
    framework_version="2.1",
    environment_definition=environment,
    compute_target="local",
    script_params={'--data-size': 3000},
    inputs=[fra_eng_ds.as_named_input('in_data')]
)

experiment = Experiment(workspace=ws, name="translator-fr-en")
run = experiment.submit(config=estimator)

run.wait_for_completion(show_output=True)