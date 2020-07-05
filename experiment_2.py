from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

environment = Environment.from_conda_specification(name="sentiment-env", file_path="experiment-env.yml")
environment.register(ws)
# environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="imdb", 
    entry_script="experiment.py", 
    compute_target="local", 
    framework_version="2.1",  
    script_params={'--n-words': 80000, '--epochs': 2},
    environment_definition=environment
    )

experiment = Experiment(workspace=ws, name="sentiment-analysis")
run = experiment.submit(config=estimator)

run.wait_for_completion(show_output=True)

run.register_model( model_name='sentiment_model',
                    model_path=f'outputs/sentiment_model.h5',
                    description='A sentiment analysis model from imdb data',
                    tags={'source': 'imdb'},
                    model_framework=Model.Framework.TENSORFLOW,
                    model_framework_version='2.2.0',
                    properties={'Accuracy': run.get_metrics()['accuracy']})