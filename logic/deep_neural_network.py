import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import shap
from logic.functions import evaluate_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def deep_neural_network(X, y, X_train, X_test, y_train, y_test):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

    y_pred = model.predict(X_test)

    y_test = np.array(y_test).reshape((-1, 1))

    results = evaluate_model(y_test, np.where(y_pred > 0.5, 1, 0))

    return results


def gradient_based_method(model, input_data, label):
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        output = model(input_data)
        loss = tf.keras.losses.binary_crossentropy(label, output)

    # Compute gradients of loss with respect to input data
    gradients = tape.gradient(loss, input_data)

    # Compute feature importance scores using gradients
    importance_scores = tf.reduce_mean(gradients, axis=0)

    return importance_scores
