from azureml.core import Environment, Experiment, Workspace, Run, Model, Datastore
import argparse
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

run = Run.get_context()
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

model_dir = "outputs"
os.makedirs(model_dir, exist_ok=True)

gpus = tf.config.experimental.list_physical_devices("GPU")

if len(gpus) > 0:
    run.log('mode', 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    run.log('mode', 'CPU')


def prepare_embedding(review):
    encoded_doc = tf.keras.preprocessing.text.one_hot(review, n_words)
    padded_doc = preprocessing.sequence.pad_sequences(
        [encoded_doc], maxlen=max_len, padding="post")
    return padded_doc


def load_data():
    # load data
    (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=n_words)

    # Pad sequences with max_len
    X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
    return (X_train, y_train), (X_test, y_test)


def build_model():
    model = models.Sequential()

    model.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


(X_train, y_train), (X_test, y_test) = load_data()
model = build_model()
model.summary()

model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

model.save(f'{model_dir}/sentiment_model.h5')

scores = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

run.log("accuracy", (scores[1] * 100))

run.complete()

run.register_model( model_name='sentiment_model',
                    model_path='outputs/sentiment_model.h5',
                    description='A sentiment analysis model from imdb data',
                    tags={'source': 'imdb'},
                    model_framework=Model.Framework.TENSORFLOW,
                    model_framework_version='2.2.0',
                    properties={'Accuracy': run.get_metrics()['accuracy']})