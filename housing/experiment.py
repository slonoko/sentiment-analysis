import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
import pandas as pd
from azureml.core import Dataset, Run, Workspace, Model
from dotnetcore2 import runtime
import argparse
import os

#runtime.version = ("18", "04", "0")
#runtime.dist = "ubuntu"
gpus = tf.config.experimental.list_physical_devices("GPU")

run = Run.get_context()
workspace = run.experiment.workspace
parser = argparse.ArgumentParser()

if len(gpus) > 0:
    run.log('mode', 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    run.log('mode', 'CPU')

model_dir = "outputs"
os.makedirs(model_dir, exist_ok=True)



parser.add_argument('--nb-steps', type=int, dest='nb_steps', default=1000)

args = parser.parse_args()

nb_steps = args.nb_steps

dataset = Dataset.get_by_name(workspace, name='ds_boston_housing')
dataframe = dataset.to_pandas_dataframe()

dataframe.head()
dataframe.describe()
dataframe.corr()

train, test = train_test_split(dataframe, test_size=0.005, random_state=0)
train, val = train_test_split(train, test_size=0.2, random_state=0)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('medv')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


feature_columns = []

# numeric cols
for header in dataframe.columns.values:
  if header != 'medv':
    feature_columns.append(feature_column.numeric_column(header))


batch_size = 64
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  layers.DenseFeatures(feature_columns),
  layers.Dense(128, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=nb_steps)

"""
def input_fct():
  dataframe = train.copy()
  labels = dataframe.pop('medv')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(32)
  return ds

def val_input_fct():
  dataframe = val.copy()
  labels = dataframe.pop('medv')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(32)
  return ds

model = tf.estimator.LinearRegressor(feature_columns=feature_columns)
model.train(input_fn=input_fct,steps=10000)
"""

model.save(f'{model_dir}/boston_housing_model.h5')
loss, accuracy = model.evaluate(test_ds)

run.log("accuracy", accuracy)

""" res = model.evaluate(input_fn=val_input_fct)

run.log("average_loss", res['average_loss'])
run.log("label_mean", res['label/mean'])
run.log("loss", res['loss'])
run.log("prediction_mean", res['prediction/mean'])
run.log("global_step", res['global_step']) """
run.complete()
