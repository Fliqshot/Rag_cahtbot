import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import time

def initialize_camera():
    # Try different camera indices
    for i in range(3):  # Try first 3 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Successfully opened camera {i}")
            return cap
        cap.release()
        time.sleep(1)  # Wait before trying next camera
    
    print("Could not initialize any camera. Please check camera permissions and connections.")
    return None

# Initialize the face detection pipeline
face_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# Initialize the emotion detection pipeline
emotion_detector = pipeline("image-classification", model="dima806/facial_emotions_image_detection", top_k=1)

cap = initialize_camera()
if cap is None:
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB for the face detector
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Convert frame to PIL Image for face detection
        pil_image = Image.fromarray(rgb_frame)
        
        # Detect faces using the Hugging Face model
        detections = face_detector(pil_image)
        
        for detection in detections:
            if detection['label'] == 'person':  # Filter for face detections
                box = detection['box']
                x, y, w, h = int(box['xmin']), int(box['ymin']), int(box['xmax'] - box['xmin']), int(box['ymax'] - box['ymin'])
                
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face region
                face = frame[y:y + h, x:x + w]
                if face.size > 0:  # Check if face region is valid
                    # Convert face to PIL Image for emotion detection
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    
                    # Detect emotion
                    emotion_result = emotion_detector(face_pil)
                    emotion = emotion_result[0]['label']
                    confidence = emotion_result[0]['score']
                    
                    # Display emotion text with confidence
                    text = f"{emotion} ({confidence:.2f})"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    except Exception as e:
        print(f"Error during face/emotion detection: {str(e)}")
        continue

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
