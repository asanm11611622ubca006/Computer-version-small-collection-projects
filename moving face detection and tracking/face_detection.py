"""
Real-time Face Detection and Tracking System
Uses OpenCV's Haar Cascade algorithm for detecting and tracking human faces
"""

import cv2
import numpy as np
import time

class FaceDetector:
    def __init__(self):
        """Initialize the face detector with Haar Cascade classifier"""
        # Load the pre-trained Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Check if webcam opened successfully
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam")
        
        # Set camera properties for better performance and quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create a named window that can be resized
        cv2.namedWindow('Real-time Face Detection and Tracking', cv2.WINDOW_NORMAL)
        # Set window to fullscreen mode
        cv2.setWindowProperty('Real-time Face Detection and Tracking', 
                            cv2.WND_PROP_FULLSCREEN, 
                            cv2.WINDOW_FULLSCREEN)
        
        # Variables for FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        print("Face Detection System Initialized")
        print("Press 'q' or 'ESC' to exit")
    
    def detect_faces(self, frame):
        """
        Detect faces in the given frame
        
        Args:
            frame: Input image frame
            
        Returns:
            faces: Array of face coordinates (x, y, w, h)
        """
        # Convert frame to grayscale (Haar Cascade works on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection in varying lighting
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with strict parameters to avoid false positives
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Standard scale for reliable detection
            minNeighbors=8,       # Higher value = more strict (fewer false positives)
            minSize=(80, 80),     # Larger minimum size to filter out small false detections
            maxSize=(400, 400),   # Maximum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def draw_face_boxes(self, frame, faces):
        """
        Draw bounding boxes around detected faces
        
        Args:
            frame: Input image frame
            faces: Array of face coordinates
            
        Returns:
            frame: Frame with drawn bounding boxes
        """
        for (x, y, w, h) in faces:
            # Draw rectangle around face with thicker border for visibility
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add label above the rectangle
            label = "Face"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background for label
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        return int(fps)
    
    def draw_info(self, frame, fps, face_count):
        """
        Draw information overlay on frame
        
        Args:
            frame: Input image frame
            fps: Current FPS
            face_count: Number of detected faces
        """
        # Draw semi-transparent background for info panel (larger for better visibility)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw FPS with larger font
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Draw face count with larger font
        cv2.putText(
            frame,
            f"Faces Detected: {face_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Add instruction text
        cv2.putText(
            frame,
            "Press 'q' or 'ESC' to exit | 'f' for fullscreen toggle",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    def run(self):
        """Main loop for face detection and tracking"""
        print("\n=== Face Detection System Running ===")
        print("Press 'f' to toggle fullscreen mode")
        
        is_fullscreen = True
        
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Detect faces in the frame
            faces = self.detect_faces(frame)
            
            # Draw bounding boxes around detected faces
            frame = self.draw_face_boxes(frame, faces)
            
            # Calculate FPS
            fps = self.calculate_fps()
            
            # Draw information overlay
            self.draw_info(frame, fps, len(faces))
            
            # Display the resulting frame
            cv2.imshow('Real-time Face Detection and Tracking', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Toggle fullscreen with 'f' key
            if key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty('Real-time Face Detection and Tracking',
                                        cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('Real-time Face Detection and Tracking',
                                        cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_NORMAL)
            
            # Exit on 'q' or 'ESC' key
            if key == ord('q') or key == 27:  # 27 is ESC key
                print("\nExiting Face Detection System...")
                break
        
        # Cleanup
        self.release()
    
    def release(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released successfully")


def main():
    """Main function to run the face detection system"""
    try:
        # Create face detector instance
        detector = FaceDetector()
        
        # Run the detection system
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your webcam is connected and not being used by another application")
        print("2. Check if OpenCV is installed correctly: pip install opencv-python")
        print("3. Verify that the Haar Cascade file is available")


if __name__ == "__main__":
    main()
