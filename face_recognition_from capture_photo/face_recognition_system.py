# --------------------------------------------------------------
# 1️⃣  IMPORTS & SET‑UP
# --------------------------------------------------------------
import cv2                               ## OpenCV: core library for image/video I/O and computer‑vision algorithms
import os                                ## OS: file‑system utilities (path handling, directory walking)
import numpy as np                       ## NumPy: efficient array handling – required by OpenCV face recognizer
from datetime import datetime            ## datetime: to generate unique timestamp filenames for unknown faces

# --------------------------------------------------------------
# 2️⃣  GLOBAL SETTINGS (Windows‑compatible paths)
# --------------------------------------------------------------
# Base folder that contains one sub‑folder per person, e.g.
# datasets/
#   ├─ alice/
#   │    ├─ face_1.jpg
#   │    └─ face_2.jpg
#   └─ bob/
#        ├─ face_1.jpg
#        └─ face_2.jpg
DATASET_DIR = os.path.join("datasets")       ## Root of the image collection
MODEL_PATH  = os.path.join("trainer.yml")    ## Where the trained LBPH model is stored
UNKNOWN_DIR = os.path.join("unknown")        ## Folder to save unknown face images

# Confidence threshold for LBPH recognizer
# LBPH returns LOWER values for BETTER matches.
# If confidence > threshold → person is UNKNOWN
# If confidence <= threshold → person is KNOWN
CONFIDENCE_THRESHOLD = 80  ## Tune this value based on your lighting/dataset quality

# --------------------------------------------------------------
# 3️⃣  HELPER: READ IMAGES & ASSIGN NUMERIC LABELS
# --------------------------------------------------------------
def load_images_and_labels(dataset_dir):
    """
    Walk through each sub‑folder of `dataset_dir`.
    * Sub‑folder name → person name (string)
    * Assign a unique integer label to each person
    * Load every image, convert to grayscale, and store it with its label
    Returns:
        faces   – list of grayscale face images (as NumPy arrays)
        labels  – list of integer labels matching `faces`
        label_id_name_map – dict {label_id: person_name}
    """
    faces = []                ## List[ndarray] – raw face crops
    labels = []               ## List[int]   – numeric label per face
    label_id_name_map = {}    ## Mapping back from numeric id → readable name
    current_label = 0

    # Iterate over each person‑folder
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue                     ## Skip stray files

        # Register the numeric label for this person
        label_id_name_map[current_label] = person_name

        # Load every image file inside the person's folder
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            # Only process common image extensions
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Read the image in grayscale (LBPH works on single‑channel images)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue                 ## Skip unreadable files

            faces.append(img)
            labels.append(current_label)

        current_label += 1               ## Increment label for next person

    return faces, labels, label_id_name_map

# --------------------------------------------------------------
# 4️⃣  TRAIN THE LBPH RECOGNIZER
# --------------------------------------------------------------
def train_recognizer():
    """
    1. Load all face images and their numeric labels.
    2. Initialise an LBPHFaceRecognizer (Local Binary Patterns Histograms).
    3. Train it on the collected data.
    4. Save the trained model to `trainer.yml`.
    5. Return the recognizer and the label‑to‑name map for later use.
    """
    print("[INFO] Loading images from dataset …")
    faces, labels, label_id_name_map = load_images_and_labels(DATASET_DIR)

    if not faces:
        raise RuntimeError("No face images found – check the `datasets` folder structure.")

    # Convert Python lists to the format expected by OpenCV (NumPy arrays)
    faces_np  = [np.array(face, dtype=np.uint8) for face in faces]
    labels_np = np.array(labels, dtype=np.int32)

    # Initialise the recogniser – requires `opencv-contrib-python`
    recogniser = cv2.face.LBPHFaceRecognizer_create()

    print("[INFO] Training recogniser …")
    recogniser.train(faces_np, labels_np)   ## Core training step

    # Persist the trained model for later use
    recogniser.write(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    return recogniser, label_id_name_map

# --------------------------------------------------------------
# 5️⃣  HELPER: SAVE UNKNOWN FACE WITH TIMESTAMP
# --------------------------------------------------------------
def save_unknown_face(face_image):
    """
    Save an unknown face image to the 'unknown' folder.
    Uses a timestamp in the filename to avoid overwriting previous images.
    """
    # Create the unknown folder if it doesn't exist
    os.makedirs(UNKNOWN_DIR, exist_ok=True)  ## exist_ok=True prevents error if folder already exists

    # Generate a unique filename using current timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")  ## Format: 2026_01_06_18_45_10_123456
    filename = f"unknown_{timestamp}.jpg"                       ## Example: unknown_2026_01_06_18_45_10_123456.jpg
    filepath = os.path.join(UNKNOWN_DIR, filename)              ## Full path: unknown/unknown_2026_01_06_18_45_10_123456.jpg

    # Save the face image to disk
    cv2.imwrite(filepath, face_image)  ## Write the grayscale face image
    print(f"[INFO] Unknown face saved: {filepath}")

# --------------------------------------------------------------
# 6️⃣  LIVE RECOGNITION (Webcam) WITH UNKNOWN FACE HANDLING
# --------------------------------------------------------------
def recognize_from_webcam(recogniser, label_id_name_map):
    """
    Open the default webcam, detect faces with Haar cascade,
    predict the person using the trained LBPH model,
    and display the name + confidence on the video feed.
    
    UNKNOWN FACE HANDLING:
    - If confidence > CONFIDENCE_THRESHOLD → face is UNKNOWN
    - Display "UNKNOWN" label on screen
    - Save the face image to the 'unknown' folder with timestamp
    """
    # Haar cascade for frontal‑face detection (bundled with OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Start video capture (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam – ensure it is connected and not used by another app.")

    # Counter to limit how often we save unknown faces (to avoid flooding the folder)
    last_save_time = datetime.now()  ## Track last save time
    save_interval_seconds = 2        ## Minimum seconds between saving unknown faces

    print("[INFO] Starting live face recognition – press 'q' to quit.")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD} (higher = unknown)")
    
    while True:
        ret, frame = cap.read()  ## Read a frame from the webcam
        if not ret:
            print("[WARN] Frame capture failed – exiting.")
            break

        # Convert to grayscale – required for both detection & recogniser
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,    ## How much the image size is reduced at each scale
            minNeighbors=5,     ## How many neighbors each candidate rectangle should have
            minSize=(60, 60)    ## Minimum possible face size
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest) from the grayscale image
            face_roi = gray[y:y+h, x:x+w]
            # Resize to the same size used during training for consistency
            face_resized = cv2.resize(face_roi, (200, 200))

            # Predict the label and obtain a confidence score
            label_id, confidence = recogniser.predict(face_resized)

            # ========================================================
            # CONFIDENCE THRESHOLD LOGIC (CRITICAL FOR UNKNOWN DETECTION)
            # ========================================================
            # LBPH returns LOWER values for BETTER matches:
            #   - confidence = 0   → perfect match
            #   - confidence = 50  → good match
            #   - confidence = 100 → poor match
            #   - confidence = 150+ → very poor match (likely unknown)
            #
            # If confidence <= CONFIDENCE_THRESHOLD → KNOWN person
            # If confidence > CONFIDENCE_THRESHOLD  → UNKNOWN person
            # ========================================================

            if confidence <= CONFIDENCE_THRESHOLD:
                # --- KNOWN FACE ---
                name = label_id_name_map.get(label_id, "Unknown")  ## Get name from label map
                display_text = f"{name} ({confidence:.1f})"        ## Display name and confidence
                box_color = (0, 255, 0)                            ## Green box for known faces
            else:
                # --- UNKNOWN FACE ---
                display_text = f"UNKNOWN ({confidence:.1f})"       ## Display UNKNOWN label
                box_color = (0, 0, 255)                            ## Red box for unknown faces

                # Save the unknown face (with rate limiting to avoid too many saves)
                time_since_last_save = (datetime.now() - last_save_time).total_seconds()
                if time_since_last_save >= save_interval_seconds:
                    save_unknown_face(face_resized)   ## Save cropped face image
                    last_save_time = datetime.now()   ## Update last save time

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Put the name/confidence label above the rectangle
            cv2.putText(frame, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Show the annotated video stream
        cv2.imshow("Live Face Recognition - Press 'q' to quit", frame)

        # Exit when the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed – exiting.")
            break

    # Release resources
    cap.release()              ## Free the webcam
    cv2.destroyAllWindows()    ## Close all OpenCV windows

# --------------------------------------------------------------
# 7️⃣  MAIN ENTRY POINT
# --------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Load (or train) the recogniser
    # If a trained model already exists we can skip retraining – this speeds up testing.
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading existing model from {MODEL_PATH}")
        recogniser = cv2.face.LBPHFaceRecognizer_create()  ## Create empty recognizer
        recogniser.read(MODEL_PATH)                         ## Load trained weights

        # We still need the label‑to‑name map; rebuild it from the dataset folder
        _, _, label_id_name_map = load_images_and_labels(DATASET_DIR)
    else:
        # No model found – train from scratch
        recogniser, label_id_name_map = train_recognizer()

    # 2️⃣ Start live webcam recognition with unknown face handling
    recognize_from_webcam(recogniser, label_id_name_map)