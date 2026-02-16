# Import required libraries
import cv2  ## OpenCV library for image/video processing and computer vision tasks
import os   ## OS library for handling directory creation and file paths

# Prompt the user for a name to label the dataset folder
user_name = input("Enter your name (used for dataset folder): ")  ## Input ensures each user has a separate folder

# Define the path where face images will be saved (Windows compatible path)
save_dir = os.path.join("datasets", user_name)  ## Creates path like 'datasets/John'
os.makedirs(save_dir, exist_ok=True)  ## Ensure the folder exists; creates intermediate directories if needed

# Load the pre-trained Haar Cascade classifier for frontal face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  ## OpenCV provides the XML file path
face_cascade = cv2.CascadeClassifier(cascade_path)  ## Initialize the classifier

# Initialize webcam video capture (0 is the default camera)
cap = cv2.VideoCapture(0)  ## Starts capturing frames from the webcam

# Counter for saved face images
count = 0  ## We'll collect at least 100 images

print("Starting face capture. Press 'q' to quit early.")
while True:
    ret, frame = cap.read()  ## Read a frame from the webcam
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  ## Convert to grayscale for detection (faster & required)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )  ## Detect faces; returns list of rectangles (x, y, w, h)

    # Iterate over detected faces (usually one)
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]  ## Crop the face area from the grayscale image
        # Optionally resize to a consistent size (e.g., 200x200)
        face_resized = cv2.resize(face_roi, (200, 200))
        # Save the face image to the dataset folder
        img_path = os.path.join(save_dir, f"face_{count+1}.jpg")
        cv2.imwrite(img_path, face_resized)  ## Write image file to disk
        count += 1
        # Draw a rectangle around the face on the original frame for visual feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Show the count on the frame
        cv2.putText(frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        break  # Process only the first detected face per frame

    # Display the frame with annotations
    cv2.imshow('Face Capture', frame)  ## Opens a window showing live video

    # Check for user key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q' pressed. Exiting.")
        break
    # Stop after collecting 100 images
    if count >= 100:
        print("Collected 100 face images. Stopping.")
        break

# Release resources
cap.release()  ## Free the webcam
cv2.destroyAllWindows()  ## Close any OpenCV windows

print(f"Dataset collection complete. Images saved in: {save_dir}")
