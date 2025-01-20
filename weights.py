import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
import os

import h5py
import numpy as np
from tensorflow.keras.models import load_model

# Open the weights file
with h5py.File('pretrained_siamese_weights.weights.h5', 'r+') as f:
    # Access the specific weights for the Dense layer
    dense_weights = f['dense/kernel'][:]
    
    # Resize the weights
    resized_weights = np.resize(dense_weights, (153664, 128))
    
    # Overwrite the resized weights in the file
    del f['dense/kernel']
    f.create_dataset('dense/kernel', data=resized_weights)


# Step 1: Define the base network
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

# Step 2: Define the Siamese Neural Network
def create_siamese_model(input_shape):
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    # Compute L1 distance
    l1_distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([embedding_a, embedding_b])
    output = Dense(1, activation='sigmoid')(l1_distance)

    return Model([input_a, input_b], output)

# Step 3: Initialize the model and create mock weights
input_shape = (105, 105, 1)  # Example input shape
siamese_model = create_siamese_model(input_shape)

# Initialize the model by running dummy data through it
dummy_input_a = tf.random.normal((1, 105, 105, 1))
dummy_input_b = tf.random.normal((1, 105, 105, 1))
_ = siamese_model([dummy_input_a, dummy_input_b])

# Save the initialized weights with the correct file extension
weights_path = "pretrained_siamese_weights.weights.h5"
siamese_model.save_weights(weights_path)
print(f"Pre-trained weights file created and saved at: {weights_path}")
