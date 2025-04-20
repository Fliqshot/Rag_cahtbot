import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import time
import sys

def get_user_permission():
    """Ask for user permission to access the camera."""
    print("\n" + "="*50)
    print("Emotion Detection System")
    print("="*50)
    print("\nThis application requires access to your camera for emotion detection.")
    print("The camera will be used to analyze your facial expressions in real-time.")
    print("\nPrivacy Note:")
    print("- The video feed is processed locally on your device")
    print("- No images or video are stored or transmitted")
    print("- You can press 'q' at any time to exit")
    
    while True:
        response = input("\nDo you want to proceed with camera access? (yes/no): ").lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            print("\nCamera access denied. Exiting...")
            return False
        else:
            print("Please enter 'yes' or 'no'")

def initialize_camera():
    """Try to initialize the camera with user feedback."""
    print("\nInitializing camera...")
    for i in range(3):  # Try first 3 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✓ Camera {i} successfully initialized")
            return cap
        cap.release()
        time.sleep(1)
    
    print("✗ Could not initialize any camera")
    print("Please check if:")
    print("1. Your camera is properly connected")
    print("2. No other application is using the camera")
    print("3. You have granted camera permissions")
    return None

def main():
    # Get user permission first
    if not get_user_permission():
        sys.exit(0)

    # Initialize the face detection classifier
    print("\nLoading face detection model...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("✓ Face detection model loaded")

    # Initialize the emotion detection pipeline
    print("\nLoading emotion detection model...")
    emotion_detector = pipeline("image-classification", model="dima806/facial_emotions_image_detection", top_k=1)
    print("✓ Emotion detection model loaded")

    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)

    print("\nStarting emotion detection...")
    print("Press 'q' to quit")
    print("-"*50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # Detect faces using Haar Cascade
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
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
            print("\nStopping emotion detection...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nEmotion detection stopped. Thank you for using the system!")

if __name__ == "__main__":
    main()