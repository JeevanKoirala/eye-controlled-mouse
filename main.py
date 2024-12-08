import cv2
import mediapipe as mp
import pyautogui
import time


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)


screen_w, screen_h = pyautogui.size()

blink_count = 0
last_blink_time = 0


def calculate_ear(eye):
    left = eye[0]
    right = eye[1]
    top = eye[2]
    bottom = eye[3]
    horizontal_distance = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
    vertical_distance = ((top.x - bottom.x) ** 2 + (top.y - bottom.y) ** 2) ** 0.5
    return vertical_distance / horizontal_distance

while True:
    ret, frame = cam.read()
    if not ret:
        raise Exception("Unable to capture video from webcam.")
    

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    
    frame_size = min(frame_w, frame_h)
    frame_resized = cv2.resize(frame, (frame_size, frame_size))
    
    
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    
    output = face_mesh.process(rgb_frame)

    image_dimensions = (frame_w, frame_h)
    
    
    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        
        
        left_eye = [landmarks[145], landmarks[159], landmarks[223], landmarks[230]]
        right_eye = [landmarks[374], landmarks[386], landmarks[443], landmarks[450]]
        
        
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        
        
        if left_ear < 0.25 and right_ear < 0.25:  
            current_time = time.time()
            if current_time - last_blink_time < 0.3:  
                blink_count += 1
                pyautogui.click()  
            last_blink_time = current_time
        
        
        for id, landmark in enumerate(landmarks[474:478]): 
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:  
                screen_x = screen_w * (landmark.x)
                screen_y = screen_h * (landmark.y)
                pyautogui.moveTo(screen_x, screen_y)  
        
    
    cv2.imshow('Eye Cursor', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()



