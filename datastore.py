from azureml.core import Dataset, Workspace
from dotnetcore2 import runtime
runtime.version = ("18", "04", "0")
runtime.dist = "ubuntu"
ws = Workspace.from_config()

default_ds = ws.get_default_datastore()

data_ref = default_ds.upload(src_dir='data',target_path='/data/files', overwrite=True, show_progress=True)
housing_ds = Dataset.Tabular.from_delimited_files(path=(default_ds,'/data/files/boston_housing.csv'))
housing_ds.register(workspace=ws, name='ds_boston_housing')