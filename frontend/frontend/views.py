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
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

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
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img
