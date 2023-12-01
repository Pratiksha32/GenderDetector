
import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained gender classification model
gender_net = cv2.dnn.readNetFromCaffe(
    'D:\JBK\python project\Age _ Gender Recongniition\gender_deploy.prototxt',
    'D:\JBK\python project\Age _ Gender Recongniition\gender_net.caffemodel'
)

# Define the function to detect faces and classify gender
def detect_and_classify(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face ROI for gender classification
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(blob)

        # Forward pass to get gender prediction
        gender_preds = gender_net.forward()

        # Get the gender label
        gender = 'Male' if gender_preds[0][0] > 0.5 else 'Female'

        # Display the gender label
        cv2.putText(frame, f'Gender: {gender}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Access the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform face detection and gender classification
    frame = detect_and_classify(frame)

    # Display the frame
    cv2.imshow('Gender Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
