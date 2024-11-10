import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import easyocr  # Importando EasyOCR

model = tf.keras.models.load_model('./object_detection.h5')
print('Model loaded Sucessfully')


path = 'images/Cars52.png'
image = load_img(path) # PIL object
image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
image1 = load_img(path,target_size=(224,224)) #converte a imagem para 255,255
image_arr_224 = img_to_array(image1)/255.0  # Convert into array and get the normalized output

# Size of the orginal image
h,w,d = image.shape
print('Height of the image =',h)
print('Width of the image =',w)




image_arr_224.shape



test_arr = image_arr_224.reshape(1,224,224,3) 
test_arr.shape




fig = px.imshow(image)
fig.update_layout(width=700, height=500,  margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 1 - TEST Image')

# Fazer a previsão com o modelo
coords = model.predict(test_arr)
print("Coordenadas normalizadas:", coords)

# Desnormalizar as coordenadas
denorm = np.array([w, w, h, h])
coords = coords * denorm
coords = coords.astype(np.int32)
print("Coordenadas desnormalizadas:", coords)

# Extrair as coordenadas
xmin, xmax, ymin, ymax = coords[0]
pt1 = (xmin, ymin)
pt2 = (xmax, ymax)

# Desenhar a caixa delimitadora na imagem
cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)

# Exibir a imagem com a caixa delimitadora
#fig = px.imshow(image)
#fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 1 - TEST Image')
#fig.show()

# 6.3 Extração da placa (região de interesse - ROI)
img = np.array(load_img(path))  # Recarregar a imagem como array
xmin, xmax, ymin, ymax = coords[0]  # Coordenadas da caixa
roi = img[ymin:ymax, xmin:xmax]  # Recortar a região de interesse (ROI)

# Exibir a imagem cortada (Cropped Image)
fig_roi = px.imshow(roi)
fig_roi.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 15 Cropped image')
fig_roi.show()


# Aplicando EasyOCR na ROI
reader = easyocr.Reader(['en'])  # Carrega o modelo OCR (inglês por padrão)
result = reader.readtext(roi)

# Exibindo os resultados da OCR
print("Texto detectado na placa:", result)

# Se você quiser apenas o texto, pode extrair o primeiro resultado assim:
if result:
    plate_text = result[0][-2]  # O segundo último item é o texto detectado
    print("Texto da placa:", plate_text)