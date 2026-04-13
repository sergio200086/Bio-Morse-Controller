import cv2
import mediapipe as mp

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
print("Starting controller, press q to exit.")

with HandLandmarker.create_from_options(opciones) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        resultados = landmarker.detect(mp_image)
        
        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                thumb_tip = hand_landmarks[4]
                index_finger_tip = hand_landmarks[8]
                middle_finger_tip = hand_landmarks[12]
                ring_finger_tip = hand_landmarks[16]
                pinky_tip = hand_landmarks[20]
                
                h_thumb, w_thumb, c_thumb = frame.shape
                x_thumb = int(thumb_tip.x * w_thumb)
                y_thumb = int(thumb_tip.y * h_thumb)
                cv2.circle(frame, (x_thumb, y_thumb), 5, (0, 255, 0), -1)


                h_index, w_index, c_index = frame.shape
                x_index = int(index_finger_tip.x * w_index)
                y_index = int(index_finger_tip.y * h_index)
                cv2.circle(frame, (x_index, y_index), 5, (0, 255, 0), -1)

                h_middle, w_middle, c_middle = frame.shape
                x_middle = int(middle_finger_tip.x * w_middle)
                y_middle = int(middle_finger_tip.y * h_middle)
                cv2.circle(frame, (x_middle, y_middle), 5, (0, 255, 0), -1)


                h_ring, w_ring, c_ring = frame.shape
                x_ring = int(ring_finger_tip.x * w_ring)
                y_ring = int(ring_finger_tip.y * h_ring)
                cv2.circle(frame, (x_ring, y_ring), 5, (0, 255, 0), -1)

                h_pinky, w_pinky, c_pinky = frame.shape
                x_pinky = int(pinky_tip.x * w_pinky)
                y_pinky = int(pinky_tip.y * h_pinky)
                cv2.circle(frame, (x_pinky, y_pinky), 5, (0, 255, 0), -1)

                cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (0, 255, 0))

        # Mostrar pantalla modo espejo
        cv2.imshow('Bio-Morse Vision', cv2.flip(frame, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()