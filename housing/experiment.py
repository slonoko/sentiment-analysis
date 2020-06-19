import shutil
import numpy as np
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers

# Step 1 : Let's import the data
dataset_csv = "boston_housing.csv"

import pandas as pd

dataframe = pd.read_csv(dataset_csv)

# Step 2 : Explore the data
dataframe.head()
dataframe.describe()
dataframe.corr()

# Step 3 : Shuffle, Scale and Spilt the data
# dependent_columns = ['rm', 'lstat', 'ptratio']
# target_columns = ['medv']
# dependent_variables = pd.DataFrame(dataset[dependent_columns])
# target_variable = pd.DataFrame(dataset[target_columns])

# from sklearn.preprocessing import StandardScaler
#
# standard_scalar = StandardScaler()
# dataframe = standard_scalar.fit_transform(dataframe)

# dataframe = pd.DataFrame(dataframe)
# dataframe.rename(columns={0: 'rm', 1: 'lstat', 2: 'ptratio', 3: 'medv'}, inplace=True)

from sklearn.model_selection import train_test_split

train, test = train_test_split(dataframe, test_size=0.005, random_state=0)
train, val = train_test_split(train, test_size=0.2, random_state=0)
print(test)

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
for header in ['rm', 'lstat', 'ptratio']:
  feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 64
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
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
          epochs=10000)

print(test)
model.predict(test_ds)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)