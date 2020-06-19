from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace, Environment, Model
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

environment = Environment.from_conda_specification(name="sentiment-env", file_path="experiment-env.yml")
environment.register(ws)
# environment = Environment.get(ws, "sentiment-env")

estimator = TensorFlow(
    source_directory="housing", 
    entry_script="experiment.py", 
    compute_target="archi-trainer", 
    framework_version="2.1",  
    script_params={'--nb-steps': 1000},
    environment_definition=environment
    )

experiment = Experiment(workspace=ws, name="housing-prices")
run = experiment.submit(config=estimator)

run.wait_for_completion(show_output=True)

run.register_model( model_name='boston_housing_model',
                    model_path=f'outputs/boston_housing_model.h5',
                    description='A Housing  model Boston in the 70s',
                    tags={'source': 'housing_boston'},
                    model_framework=Model.Framework.TENSORFLOW,
                    model_framework_version='2.2.0',
                    properties={'accuracy': run.get_metrics()['accuracy']})