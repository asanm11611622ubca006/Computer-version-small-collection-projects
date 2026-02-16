# Quick Start Guide - Real-Time Object Detection

## 🚀 Quick Setup (3 Steps)

### Step 1: Install Python Libraries

Open Command Prompt in this directory and run:

```bash
pip install -r requirements.txt
```

### Step 2: Download Model Files

Run the download script:

```bash
powershell -ExecutionPolicy Bypass -File download_models.ps1
```

**OR** manually download:

- [MobileNetSSD_deploy.prototxt](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt)
- [MobileNetSSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel)

Place both files in this directory.

### Step 3: Run the Program

```bash
python object_detection.py
```

Press **'q'** to quit.

---

## 📖 What This Program Does

✅ Opens your webcam  
✅ Detects objects in real-time  
✅ Draws bounding boxes around detected objects  
✅ Shows object names and confidence percentages  
✅ Displays FPS (frames per second)  
✅ Works with 20 object classes (person, car, dog, bottle, chair, etc.)  

---

## 🎯 Detectable Objects

**People & Animals**: person, bird, cat, dog, horse, sheep, cow  
**Vehicles**: aeroplane, bicycle, boat, bus, car, motorbike, train  
**Objects**: bottle, chair, diningtable, pottedplant, sofa, tvmonitor  

---

## ⚙️ Settings

**Confidence Threshold**: 0.5 (50%)  
Only objects detected with >50% confidence are shown.

To change this, edit `object_detection.py`:

```python
CONFIDENCE_THRESHOLD = 0.5  # Change to 0.3 for more detections
```

---

## 🐛 Common Issues

**"Could not open webcam"**  
→ Close other apps using the camera  
→ Try changing camera index in code: `cv2.VideoCapture(1)`

**"FileNotFoundError"**  
→ Make sure model files are downloaded  
→ Check file names match exactly

**Low FPS**  
→ Close other programs  
→ Increase confidence threshold

**No detections**  
→ Lower confidence threshold  
→ Improve lighting  
→ Move objects closer to camera

---

## 📁 Required Files

```
✓ object_detection.py
✓ MobileNetSSD_deploy.prototxt
✓ MobileNetSSD_deploy.caffemodel
✓ requirements.txt
```

---

For detailed documentation, see **README.md**
