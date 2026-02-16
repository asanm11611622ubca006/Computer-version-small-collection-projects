# Real-time Face Detection and Tracking System

A beginner-friendly computer vision project that detects and tracks human faces in real-time using OpenCV's Haar Cascade algorithm.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Real-time Face Detection**: Detects faces instantly from webcam feed
- **Multi-face Support**: Tracks multiple faces simultaneously
- **Bounding Box Visualization**: Clear green boxes around detected faces
- **Performance Monitoring**: Live FPS counter display
- **Face Counter**: Shows number of faces currently detected
- **Optimized Performance**: Smooth tracking with minimal lag
- **Easy Exit**: Simple keyboard controls (Q or ESC)

## 📋 Requirements

- Python 3.7 or higher
- Webcam/Camera
- OpenCV
- NumPy

## 🚀 Installation

1. **Clone or download this repository**

2. **Install required packages**:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python numpy
```

## 💻 Usage

Run the face detection system:

```bash
python face_detection.py
```

### Keyboard Controls

- **Q** or **ESC**: Exit the application

## 🔧 How It Works

### Haar Cascade Algorithm

This project uses the **Haar Cascade Classifier**, a machine learning-based approach for object detection:

1. **Preprocessing**: Converts frames to grayscale and applies histogram equalization
2. **Feature Detection**: Uses Haar-like features to identify face patterns
3. **Cascade Classification**: Multiple stages of classifiers filter out non-face regions
4. **Bounding Box**: Draws rectangles around detected faces

### Key Components

- **FaceDetector Class**: Main class handling detection logic
- **detect_faces()**: Performs face detection using Haar Cascade
- **draw_face_boxes()**: Visualizes detected faces with bounding boxes
- **calculate_fps()**: Monitors real-time performance
- **draw_info()**: Displays FPS and face count overlay

## 🎓 Learning Objectives

This project is perfect for learning:

- Computer vision fundamentals
- OpenCV library usage
- Real-time video processing
- Object detection algorithms
- Python class-based programming
- Performance optimization techniques

## ⚙️ Configuration

You can adjust detection parameters in `face_detection.py`:

```python
faces = self.face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,      # Adjust for detection sensitivity
    minNeighbors=5,       # Higher = fewer false positives
    minSize=(30, 30),     # Minimum face size in pixels
)
```

### Parameter Guide

- **scaleFactor**: Lower values (1.05-1.1) = more accurate but slower
- **minNeighbors**: Higher values (4-6) = fewer false detections
- **minSize**: Increase to ignore small/distant faces

## 🐛 Troubleshooting

### Webcam not opening

- Ensure no other application is using the webcam
- Try changing camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

### Poor detection accuracy

- Ensure good lighting conditions
- Face the camera directly
- Adjust `scaleFactor` and `minNeighbors` parameters

### Low FPS

- Reduce camera resolution in the code
- Close other resource-intensive applications
- Update graphics drivers

## 📚 Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Haar Cascade Explained](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
- [Computer Vision Tutorials](https://opencv-python-tutroals.readthedocs.io/)

## 🔮 Future Enhancements

- Add face recognition capabilities
- Implement eye detection
- Add smile detection
- Save detected faces to files
- Add emotion detection
- Implement face tracking with Kalman filters

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

Created as a beginner-friendly computer vision learning project.

---

**Happy Face Detecting! 😊**
