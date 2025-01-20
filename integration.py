import cv2
import numpy as np
from siamese_nn import create_siamese_model, generate_embedding  # Import SNN functions

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for layer in siamese_model.layers:
    if layer.name == 'dense':  # Skip the problematic layer
        continue
    layer.trainable = False  # Freeze other layers
model.load_weights('pretrained_siamese_weights.weights.h5', by_name=True, skip_mismatch=True)

from tensorflow.keras.models import load_model
import h5py
import numpy as np

# Open the .h5 weight file
with h5py.File('pretrained_siamese_weights.weights.h5', 'r') as f:
    # Extract weights for the dense layer
    dense_weights = f['dense/kernel:0'][:]  # Replace 'dense' with the actual layer name if different

# Resize weights to match the expected shape
resized_weights = np.resize(dense_weights, (153664, 128))  # Ensure correct shape

# Assign the resized weights to the target layer
model.layers[0].input_shape = (None, height, width, 1)

model.layers[-1].set_weights([resized_weights, np.zeros(128)])  # Adjust bias if needed


# Function to preprocess face image
def preprocess_face(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (105, 105))  # Match input shape
    normalized_face = resized_face / 255.0  # Normalize pixel values
    return np.expand_dims(normalized_face, axis=-1)

# Start video capture
cap = cv2.VideoCapture(0)
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Detect faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Crop the face
        processed_face = preprocess_face(face)
        
        # Generate embedding for the detected face
        embedding = generate_embedding(siamese_model, processed_face)
        
        # Placeholder: Compare with database embeddings (to be implemented)
        # Example: database_embedding = ...
        # similarity = cosine_similarity(embedding, database_embedding)
        # if similarity > THRESHOLD: flag_alert()

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Face Detection and SNN', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
