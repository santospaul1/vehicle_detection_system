import cv2

# Load the pre-trained Haar Cascade classifier for vehicle detection
# Specify the full path to the haarcascade_car.xml file
cascade_path = 'haarcascade_car.xml'
vehicle_cascade = cv2.CascadeClassifier(cascade_path)

# Open the video file (replace 'video.mp4' with your video file)
cap = cv2.VideoCapture('car.mp4')

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform vehicle detection
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Vehicle Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
