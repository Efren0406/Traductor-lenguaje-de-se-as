# # myapp/views.py
# from django.shortcuts import render
# from django.http import JsonResponse
# import base64
# from django.core.files.base import ContentFile
# from django.views.decorators.csrf import csrf_exempt

# def index(request):
#     return render(request, 'index.html')

# @csrf_exempt
# def upload_image(request):
#     if request.method == 'POST':
#         image_data = request.POST.get('image')
#         format, imgstr = image_data.split(';base64,')
#         ext = format.split('/')[-1]
#         data = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)
#         # Aquí puedes procesar la imagen 'data' como quieras
#         return JsonResponse({'status': 'ok'})
#     return JsonResponse({'status': 'error'})

# myapp/views.py
import cv2
import numpy as np
import base64
import mediapipe as mp
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,250,125), thickness=1, circle_radius=1)
                              ) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,125), thickness=2, circle_radius=2)
                              ) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121,44,125), thickness=1, circle_radius=1)
                              ) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,250,125), thickness=1, circle_radius=1)
                              ) # Draw right hand connections

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('action.h5')

colors = [(245,117,16), (117,245,16), (16,117,245)]

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        img_data = base64.b64decode(imgstr)
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # Procesar la imagen con OpenCV
        processed_img = process_image(img)
        
        # Convertir la imagen procesada de nuevo a base64
        _, buffer = cv2.imencode('.png', processed_img)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JsonResponse({'status': 'ok', 'processed_image': processed_img_base64})
    
    return JsonResponse({'status': 'error'})

def process_image(img):
    # Aquí puedes realizar el procesamiento de la imagen con OpenCV
    # Ejemplo: convertir la imagen a escala de grises
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Make detections
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(img, holistic)

    # Draw landmarks
    draw_styled_landmarks(image, results)
    return image
