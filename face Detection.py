import cv2
import numpy as np

def detect_faces(frame, face_cascade):
    """
    Detect faces in a given frame using a Haar Cascade Classifier.
    Args:
        frame (numpy.ndarray): The input frame from video feed.
        face_cascade (cv2.CascadeClassifier): Preloaded Haar cascade for face detection.

    Returns:
        List of bounding boxes for detected faces.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def main():
    # Load pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Detect faces
        faces = detect_faces(frame, face_cascade)

        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Real-Time Face Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
