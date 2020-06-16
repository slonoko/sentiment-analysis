from azureml.core import Workspace
import json
import numpy as np
from azureml.core.model import Model
from azureml.core import Run
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing

ws = Workspace.from_config()

max_len = 500
n_words = 5000
dim_embedding = 32
EPOCHS = 2
BATCH_SIZE = 128
model = None

def build_model():
    m = models.Sequential()

    m.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
    m.add(layers.Flatten())
    m.add(layers.Dense(16, activation='relu'))
    m.add(layers.Dense(16, activation='relu'))
    m.add(layers.Dense(1, activation='sigmoid'))

    m.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return m

def prepare_embedding(review):
    encoded_doc = tf.keras.preprocessing.text.one_hot(review, n_words)
    padded_doc = preprocessing.sequence.pad_sequences([encoded_doc], maxlen=max_len, padding="post")
    return padded_doc

def run(raw_data):
    data = prepare_embedding(json.loads(raw_data)['data'])
    predictions = model.predict(data)
    return predictions

model_path = Model.get_model_path("sentiment_model",_workspace=ws)
model = build_model()
model.load_weights(model_path)

print(json.dumps(run('{"data":"i am very happy with this movie"}').tolist()))