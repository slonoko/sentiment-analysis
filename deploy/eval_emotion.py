import json
import numpy as np
from azureml.core.model import Model
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--max-len', type=int, dest='max_len', default=500)
parser.add_argument('--n-words', type=int, dest='n_words', default=5000)
parser.add_argument('--dim-embedding', type=int, dest='dim_embedding', default=32)
parser.add_argument('--epochs', type=int, dest='epochs', default=5)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)

args = parser.parse_args()

max_len = args.max_len
n_words = args.n_words
dim_embedding = args.dim_embedding
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
model = None

def init():
    global model
    model_path = Model.get_model_path("sentiment_model")
    model = build_model()
    model.load_weights(model_path)
    
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
