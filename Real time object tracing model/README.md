# Real-Time Object Detection using MobileNetSSD

A Python project for real-time object recognition using a pre-trained MobileNetSSD model with OpenCV.

## 📋 Project Overview

This project implements real-time object detection using:
- **OpenCV DNN module** for deep learning inference
- **MobileNetSSD** pre-trained model (Caffe format)
- **Webcam feed** for real-time detection
- **Visual feedback** with bounding boxes, labels, and confidence scores

## 🎯 Detectable Objects

The MobileNetSSD model can detect 20 different object classes:
- **People & Animals**: person, bird, cat, dog, horse, sheep, cow
- **Vehicles**: aeroplane, bicycle, boat, bus, car, motorbike, train
- **Furniture & Objects**: bottle, chair, diningtable, pottedplant, sofa, tvmonitor

## 📦 Required Libraries

Install the following Python libraries:

```bash
pip install opencv-python numpy
```

### Library Details:
- **opencv-python** (cv2): Computer vision library for image processing and DNN module
- **numpy**: Numerical computing library for array operations

## 📁 Project Structure

```
Real time object tracing model/
│
├── object_detection.py              # Main Python script
├── MobileNetSSD_deploy.prototxt     # Model architecture (required)
├── MobileNetSSD_deploy.caffemodel   # Pre-trained weights (required)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔧 Setup Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Model Files

You need to download the MobileNetSSD model files and place them in the project directory:

**Option 1: Direct Download**
- Download from [MobileNetSSD GitHub Repository](https://github.com/chuanqi305/MobileNet-SSD)
- Files needed:
  - `MobileNetSSD_deploy.prototxt`
  - `MobileNetSSD_deploy.caffemodel`

**Option 2: Using wget (if available)**
```bash
# Download prototxt file
wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt

# Download caffemodel file (large file ~23MB)
wget https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel
```

### Step 3: Verify Setup
Ensure your project directory contains:
- ✅ `object_detection.py`
- ✅ `MobileNetSSD_deploy.prototxt`
- ✅ `MobileNetSSD_deploy.caffemodel`

## 🚀 How to Run

1. **Open Command Prompt/Terminal** in the project directory

2. **Run the script**:
```bash
python object_detection.py
```

3. **Using the application**:
   - The webcam will start automatically
   - Objects will be detected in real-time with bounding boxes
   - Each detection shows the class label and confidence percentage
   - FPS (Frames Per Second) is displayed in the top-left corner
   - Object count is shown below the FPS

4. **Exit the program**:
   - Press **'q'** key to quit safely

## 🎨 Features

### ✅ Real-Time Detection
- Live webcam feed processing
- Instant object recognition

### ✅ Visual Feedback
- **Bounding boxes**: Color-coded rectangles around detected objects
- **Class labels**: Object name with confidence percentage
- **FPS counter**: Real-time performance monitoring
- **Detection count**: Number of objects currently detected

### ✅ Confidence Filtering
- Threshold set to **0.5 (50%)**
- Only high-confidence detections are displayed
- Reduces false positives

### ✅ Performance Optimized
- Efficient MobileNet architecture
- Suitable for real-time applications
- Works on CPU (no GPU required)

## 🔍 How It Works

### Step-by-Step Explanation:

#### 1. **Model Loading**
```python
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
```
- Loads the pre-trained MobileNetSSD model
- Uses Caffe framework format
- Initializes the neural network

#### 2. **Webcam Initialization**
```python
cap = cv2.VideoCapture(0)
```
- Opens the default webcam (index 0)
- Captures video frames continuously

#### 3. **Frame Preprocessing**
```python
blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
```
- Resizes frame to 300x300 pixels (model input size)
- Normalizes pixel values (scaling and mean subtraction)
- Creates a "blob" (4D tensor) for the neural network

#### 4. **Object Detection**
```python
net.setInput(blob)
detections = net.forward()
```
- Feeds the preprocessed image to the network
- Performs forward pass through the neural network
- Returns detection results (bounding boxes, classes, confidences)

#### 5. **Post-Processing**
```python
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > CONFIDENCE_THRESHOLD:
        # Draw bounding box and label
```
- Filters detections by confidence threshold (0.5)
- Extracts bounding box coordinates
- Identifies object class
- Draws visualization on frame

#### 6. **Display & FPS Calculation**
```python
fps = frame_count / elapsed_time
cv2.imshow("Real-Time Object Detection", frame)
```
- Calculates frames per second
- Displays annotated frame
- Updates in real-time

## ⚙️ Configuration

You can modify these parameters in `object_detection.py`:

```python
# Confidence threshold (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5  # Increase for fewer, more confident detections

# Webcam index
cap = cv2.VideoCapture(0)  # Change to 1, 2, etc. for other cameras

# Model input size (default: 300x300)
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), ...)
```

## 🐛 Troubleshooting

### Issue: "Could not open webcam"
**Solution**: 
- Check if webcam is connected
- Try changing camera index: `cv2.VideoCapture(1)`
- Close other applications using the webcam

### Issue: "FileNotFoundError: MobileNetSSD_deploy.caffemodel"
**Solution**:
- Ensure model files are in the same directory as the script
- Verify file names match exactly (case-sensitive)
- Re-download model files if corrupted

### Issue: Low FPS
**Solution**:
- Close other resource-intensive applications
- Reduce frame resolution
- Increase confidence threshold to reduce processing

### Issue: No objects detected
**Solution**:
- Lower the confidence threshold (e.g., 0.3)
- Ensure good lighting conditions
- Move objects closer to the camera
- Check if object class is in the supported list

## 📊 Performance

- **FPS**: 15-30 FPS on modern CPUs
- **Latency**: ~30-60ms per frame
- **Accuracy**: ~70-80% mAP (mean Average Precision)
- **Model Size**: ~23MB

## 🔒 System Requirements

- **OS**: Windows 10/11 (also compatible with Linux/macOS)
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Webcam**: Any USB or built-in camera
- **Processor**: Intel i3 or equivalent (i5+ recommended)

## 📝 Code Structure

### Main Components:

1. **`load_model()`**: Loads the pre-trained MobileNetSSD model
2. **`detect_objects()`**: Performs object detection on a frame
3. **`draw_detections()`**: Draws bounding boxes and labels
4. **`display_fps()`**: Shows FPS on screen
5. **`main()`**: Main loop for real-time detection

### Key Variables:

- **`CLASSES`**: List of 21 object classes (including background)
- **`COLORS`**: Random colors for each class
- **`CONFIDENCE_THRESHOLD`**: Minimum confidence for detection (0.5)

## 🎓 Learning Resources

### Understanding the Code:
- **OpenCV DNN Module**: [OpenCV DNN Tutorial](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
- **MobileNet Architecture**: Lightweight CNN for mobile/embedded vision
- **SSD (Single Shot Detector)**: Real-time object detection algorithm

### Key Concepts:
- **Blob**: Pre-processed image in 4D format (batch, channels, height, width)
- **Forward Pass**: Running input through the neural network
- **Confidence Score**: Probability that detection is correct (0.0 to 1.0)
- **Bounding Box**: Rectangle coordinates (x1, y1, x2, y2)

## 🚀 Future Enhancements

Potential improvements:
- [ ] Add video file input option
- [ ] Save detected frames to disk
- [ ] Implement object tracking across frames
- [ ] Add sound alerts for specific objects
- [ ] Create GUI with Tkinter/PyQt
- [ ] Support for custom trained models
- [ ] Multi-camera support

## 📄 License

This project uses the MobileNetSSD model which is available under the Caffe Model Zoo license.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are correctly placed

---

**Happy Object Detecting! 🎯**
