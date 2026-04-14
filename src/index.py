import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyautogui
import time

print("\n✓ Loading model...")
try:
    with open('gesto_modelo.pkl', 'rb') as f:
        modelo, scaler = pickle.load(f)
except FileNotFoundError:
    print("ERROR: gesto_modelo.pkl not found!")
    print("Run entrenar_modelo.py first")
    exit()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='src/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

cap = cv2.VideoCapture(0)
print("Starting controller, press q to exit.")


last_action_time = 0
action_cooldown = 0.3

screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

with HandLandmarker.create_from_options(opciones) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        resultados = landmarker.detect(mp_image)
        
        h,w,c = frame.shape
        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
                landmarks_data = []
                for landmark in hand_landmarks:
                    landmarks_data.append([landmark.x, landmark.y])
                
                X = np.array(landmarks_data).flatten().reshape(1, -1)
                X = scaler.transform(X)

                prediccion = modelo.predict(X)[0]
                confianza = modelo.predict_proba(X).max()

                cv2.putText(frame, f'Gesture: {prediccion}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {confianza:.0%}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if confianza > 0.7:
                    current_time = time.time()
                    if current_time - last_action_time > action_cooldown:
                        if prediccion == "signal":
                            print(f"✓ SIGNAL detected ({confianza:.0%})")
                            index_pos = hand_landmarks[8]
                            x_cam = index_pos.x * w
                            y_cam = index_pos.y * h
                            x_screen = int(x_cam * screen_width / w)
                            y_screen = int(y_cam * screen_height / h)
                            print(f"Position x: {x_screen}, y: {y_screen}")
                            pyautogui.moveTo(x_screen, y_screen)
                            last_action_time = current_time
                        elif prediccion == "click":
                            print(f"✓ CLICK detected ({confianza:.0%})")
                            pyautogui.click()
                            last_action_time = current_time
                        elif prediccion == "open_hand":
                            print(f"✓ OPEN HAND detected ({confianza:.0%})")
                            pyautogui.press('space')
                            last_action_time = current_time
        else:
            cv2.putText(frame, 'No hand detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Bio-Morse Vision', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()