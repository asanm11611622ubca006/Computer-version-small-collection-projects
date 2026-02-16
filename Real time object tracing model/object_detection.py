"""
Real-Time Object Detection using MobileNetSSD
This script performs real-time object detection using a pre-trained MobileNetSSD model
with OpenCV's DNN module on webcam feed.
"""

import cv2
import numpy as np
import time

# Define the class labels that MobileNetSSD can detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Generate random colors for each class for visualization
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Set confidence threshold - only detections above this will be displayed
CONFIDENCE_THRESHOLD = 0.5

def load_model():
    """
    Load the pre-trained MobileNetSSD model from disk
    Returns:
        net: The loaded neural network model
    """
    print("[INFO] Loading model...")
    
    # Path to the model files (assumed to be in the same directory)
    prototxt_path = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    
    # Load the model using OpenCV's DNN module
    # readNetFromCaffe() loads Caffe models
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    print("[INFO] Model loaded successfully!")
    return net

def detect_objects(frame, net):
    """
    Perform object detection on a single frame
    
    Args:
        frame: Input image/frame from webcam
        net: Pre-trained neural network model
    
    Returns:
        frame: Frame with bounding boxes and labels drawn
        detections: Detection results from the model
    """
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Create a blob from the frame
    # Blob is a pre-processed image that the neural network expects
    # Parameters: image, scalefactor, size, mean subtraction values, swapRB, crop
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),  # Resize to 300x300 (MobileNetSSD input size)
        0.007843,                        # Scale factor (1/127.5)
        (300, 300),                      # Target size
        127.5,                           # Mean subtraction value
        swapRB=False,                    # Don't swap Red and Blue channels
        crop=False                       # Don't crop the image
    )
    
    # Pass the blob through the network to get detections
    net.setInput(blob)
    detections = net.forward()
    
    return detections, h, w

def draw_detections(frame, detections, h, w):
    """
    Draw bounding boxes and labels on detected objects
    
    Args:
        frame: Input frame
        detections: Detection results from the model
        h: Frame height
        w: Frame width
    
    Returns:
        frame: Frame with drawn bounding boxes and labels
        detection_count: Number of objects detected
    """
    detection_count = 0
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (probability) of the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring confidence > threshold
        if confidence > CONFIDENCE_THRESHOLD:
            detection_count += 1
            
            # Extract the index of the class label
            idx = int(detections[0, 0, i, 1])
            
            # Get the class label name
            label = CLASSES[idx]
            
            # Compute the (x, y) coordinates of the bounding box
            # Detections are normalized, so multiply by frame dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Get the color for this class
            color = COLORS[idx]
            
            # Draw the bounding box around the detected object
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Create label text with class name and confidence
            label_text = f"{label}: {confidence * 100:.2f}%"
            
            # Calculate label background size
            (label_w, label_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Ensure label doesn't go above frame
            label_y = max(startY, label_h + 10)
            
            # Draw filled rectangle for label background
            cv2.rectangle(
                frame,
                (startX, label_y - label_h - 10),
                (startX + label_w, label_y),
                color,
                -1  # Filled rectangle
            )
            
            # Draw the label text
            cv2.putText(
                frame,
                label_text,
                (startX, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                2
            )
    
    return frame, detection_count

def display_fps(frame, fps):
    """
    Display FPS on the frame
    
    Args:
        frame: Input frame
        fps: Frames per second value
    
    Returns:
        frame: Frame with FPS displayed
    """
    fps_text = f"FPS: {fps:.2f}"
    
    # Draw FPS on top-left corner with background
    cv2.rectangle(frame, (10, 10), (150, 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        fps_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),  # Green text
        2
    )
    
    return frame

def main():
    """
    Main function to run real-time object detection
    """
    print("[INFO] Starting Real-Time Object Detection...")
    print(f"[INFO] Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"[INFO] Detectable Classes: {', '.join(CLASSES[1:])}")  # Skip 'background'
    
    # Load the pre-trained model
    net = load_model()
    
    # Initialize webcam (0 is the default camera)
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    
    print("[INFO] Webcam started successfully!")
    print("[INFO] Press 'q' to quit")
    
    # Variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Main loop for real-time detection
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print("[ERROR] Failed to grab frame!")
            break
        
        # Perform object detection
        detections, h, w = detect_objects(frame, net)
        
        # Draw bounding boxes and labels
        frame, detection_count = draw_detections(frame, detections, h, w)
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 1:  # Update FPS every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on frame
        frame = display_fps(frame, fps)
        
        # Display detection count
        count_text = f"Objects Detected: {detection_count}"
        cv2.rectangle(frame, (10, 50), (250, 80), (0, 0, 0), -1)
        cv2.putText(
            frame,
            count_text,
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),  # Yellow text
            2
        )
        
        # Display the frame
        cv2.imshow("Real-Time Object Detection - MobileNetSSD", frame)
        
        # Check for 'q' key press to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting...")
            break
    
    # Cleanup
    print("[INFO] Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program terminated successfully!")

if __name__ == "__main__":
    main()
