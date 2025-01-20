import tensorflow as tf
import numpy as np
from keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras import backend as K

# Define the Siamese Neural Network
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def build_siamese_network(input_shape):

    input = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(7, 7), activation='relu')(input)  # Initialize `x`
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Now use `x`
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    # Wrap the input and output into a Model
    model = Model(inputs=input, outputs=x)
    return model


# Define the distance function for embeddings
def euclidean_distance(vectors):
    """
    Compute the Euclidean distance between two vectors.
    Args:
        vectors (tuple): A pair of tensors.

    Returns:
        Tensor: The Euclidean distance.
    """
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

from keras.models import Model
from keras.layers import Input, Subtract, Lambda
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dense
import keras.backend as K

def create_siamese_model(input_shape):
    # Build the base network
    base_network = build_siamese_network(input_shape)
    
    # Define two inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # Process inputs through the base network
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Compute the L1 distance with specified output shape
    l1_distance = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]),
        output_shape=lambda shapes: (shapes[0][0], 128)  # Explicitly define the output shape
    )([processed_a, processed_b])
    
    # Add a dense layer to make the final prediction
    output = Dense(1, activation='sigmoid')(l1_distance)
    
    # Create the Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)
    
    return siamese_model




# Example of generating embeddings
def generate_embedding(model, image):
    """
    Generate an embedding for a given image.
    Args:
        model (Model): The pre-trained Siamese Network.
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The embedding.
    """
    return model.predict(np.expand_dims(image, axis=0))[0]

if __name__ == "__main__":
    input_shape = (224, 224, 3)  # Example input shape, replace with your actual shape

# Create the siamese model
siamese_model = create_siamese_model(input_shape)

# Load weights into the model
siamese_model.load_weights('pretrained_siamese_weights.weights.h5')

   
