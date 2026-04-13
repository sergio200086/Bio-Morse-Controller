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

index_array = [4, 8, 12, 16, 20]

def finger_distance(finger2, finger1):
    x1, y1 = finger2
    x2, y2 = finger1
    return ((x2-x1)**2 + (y2-y1)**2)**0.5



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
                h,w,c = frame.shape
                
                for finger in index_array:
                    landmark = hand_landmarks[finger]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                x_thumb = int(thumb_tip.x * w)
                y_thumb = int(thumb_tip.y * h)
                x_index = int(index_tip.x * w)
                y_index = int(index_tip.y * h)

                cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (0, 255, 0))
                if finger_distance((x_thumb, y_thumb), (x_index, y_index)) < 10:
                    print("click")

        frame_flipped = cv2.flip(frame,1)
        cv2.imshow('Bio-Morse Vision', frame_flipped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()