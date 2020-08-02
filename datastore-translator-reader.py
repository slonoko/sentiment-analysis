from azureml.core import Dataset, Workspace, Environment
from dotnetcore2 import runtime
runtime.version = ("18", "04", "0")
runtime.dist = "ubuntu"
ws = Workspace.from_config()

environment = Environment.from_conda_specification(name="sentiment-env", file_path="experiment-env.yml")
environment.register(ws)

fra_eng_ds = ws.datasets['fra-eng-translation']
dataframe = fra_eng_ds.to_pandas_dataframe()
