import cv2
import numpy as np
import base64
import mediapipe as mp
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache

executor = ThreadPoolExecutor(max_workers=4)
mp_holistic = mp.solutions.holistic

actions = np.array(['hello', 'thanks', 'iloveyou'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('action.h5')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

sequence = []

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def index(request):
    return render(request, 'index.html')

@csrf_exempt
async def upload_image(request):
    global sequence
    if request.method == 'POST':
        image_data = request.POST.get('image')
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        img_data = base64.b64decode(imgstr)
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        cache_key = f"processed_img_{hash(img_data)}"
        cached_img = cache.get(cache_key)
        
        if cached_img:
            processed_img_base64 = cached_img
            prediction = ""
        else:
            image, results = await process_img(img)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                prediction = actions[np.argmax(res)]
            else:
                prediction = ""

            _, buffer = cv2.imencode('.png', image)
            processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            cache.set(cache_key, processed_img_base64, timeout=60*15)
        
        return JsonResponse({'status': 'ok', 'processed_image': processed_img_base64, 'translation': prediction})
    
    return JsonResponse({'status': 'error'})

async def process_img(img):
    loop = asyncio.get_event_loop()
    image, results = await loop.run_in_executor(executor, mediapipe_detection, img, mp_holistic.Holistic())
    return image, results

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(upload_image())
