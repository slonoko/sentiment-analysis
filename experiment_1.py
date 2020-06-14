from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

ws = Workspace.from_config()
ds = ws.get_default_datastore()

run_config = RunConfiguration()
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
run_config.environment.python.user_managed_dependencies = False
run_config.environment.python.conda_dependencies=CondaDependencies.create(pip_packages=['azureml-sdk[notebooks,automl,explain]'])

# environment = Environment.from_conda_specification(name="sentiment-env", file_path="experiment-env.yml")
# environment.register(ws)
environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="experiment", 
    entry_script="experiment.py", 
    framework_version="2.1",
    conda_packages=["python=3.7.4","tensorflow","tensorflow-datasets"],
    pip_packages=["azureml-sdk[notebooks,automl,explain]"],
    compute_target="local"
    )

model_step = EstimatorStep(name="training model", estimator=estimator, compute_target="dummy", estimator_entry_script_arguments=['--n-words', 5000, '--epochs', 2])
# register_step = PythonScriptStep(name="register pipeline", source_directory="sentiment_analysis", script_name="registration.py", compute_target="dummy", runconfig=run_config)
# register_step.run_after(model_step)

sentiment_pipe = Pipeline(workspace = ws, steps=[model_step])
sentiment_pipe.validate()

experiment = Experiment(workspace=ws, name="sentiment-analysis")
run = experiment.submit(config=sentiment_pipe)

run.wait_for_completion(show_output=True)

ds.upload('outputs/sentiment_model.h5','models',overwrite=True, show_progress=True)