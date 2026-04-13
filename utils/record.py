import pickle
import os
import cv2
import numpy as np
import mediapipe as mp

import os

if os.path.exists('gestos_dataset.pkl'):
    print("✓ Loading existing dataset...")
    with open('gestos_dataset.pkl', 'rb') as f:
        gestures = pickle.load(f)
else:
    print("✓ Creating new dataset...")
    gestures = {
        "click": [],
        "fist": [],
        "signal": [],
        "open_hand": [],
    }

recording = False
currentGesture = None

print("=" * 50)
print("GESTURE RECORDER")
print("=" * 50)
print("\nInstructions:")
print("  'r' = Start recording")
print("  's' = Stop recording")
print("  'q' = Exit and save")
print("\nSteps:")
print("  1. Press 'r'")
print("  2. Perform the gesture 50-100 times (quickly)")
print("  3. Press 's'")
print("  4. Change gesto_actual in the code")
print("  5. Repeat for each gesture")
print("  6. Press 'q' to save")
print("=" * 50)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='src/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(opciones) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        results = landmarker.detect(mp_image) # all hand points

        h, w, c = frame.shape
        frame_flipped = cv2.flip(frame, 1)
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:

                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x,y), 3, (0,255,0), -1)

                if recording and currentGesture:
                    landmarks_data =[]
                    for landmark in hand_landmarks:
                        landmarks_data.append([landmark.x, landmark.y])
                    
                    gestures[currentGesture].append(landmarks_data)

                    count = len(gestures[currentGesture])
                    cv2.putText(frame, f'Recording {currentGesture}: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        else:
            cv2.putText(frame, 'Hand not detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        

        if recording:
            cv2.putText(frame, f'RECORDING: {currentGesture}', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'press r to start recording', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Gestures recorder', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            recording = True
            currentGesture = "signal" 
            print(f"\n✓ Starting recording: {currentGesture}")
        
        elif key == ord('s'):
            grabando = False
            if currentGesture:
                count = len(gestures[currentGesture])
                print(f"✓ Saved: {currentGesture} ({count} examples)")
        
        elif key == ord('q'):
            print("\n✓ leaving...")
            break

with open('gestos_dataset.pkl', 'wb') as f:
    pickle.dump(gestures, f)
 
print("\n" + "=" * 50)
print("Summary:")
for gesture_name, examples in gestures.items():
    print(f"  {gesture_name}: {len(examples)} examples")
print("=" * 50)
print("✓ dataset saved in: gestos_dataset.pkl")
print("=" * 50)
 
cap.release()
cv2.destroyAllWindows()