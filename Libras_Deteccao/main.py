import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model

video_path = 'Video/video3.mp4'
cap = cv2.VideoCapture(video_path)

hands = mp.solutions.hands.Hands(max_num_hands=1)
classes = ['A', 'E', 'B', 'C', 'D']
model = load_model('models/keras_model_v2.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
text_output = ""

while True:
    success, img = cap.read()
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints is not None:
        for hand in handsPoints:
            x_max, y_max, x_min, y_min = 0, 0, w, h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max = max(x_max, x)
                x_min = min(x_min, x)
                y_max = max(y_max, y)
                y_min = min(y_min, y)
            cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

            try:
                imgCrop = img[y_min-50:y_max+50, x_min-50:x_max+50]
                imgCrop = cv2.resize(imgCrop, (224, 224))
                imgArray = np.asarray(imgCrop)
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                indexVal = np.argmax(prediction)
                cv2.putText(img, classes[indexVal], (x_min-50, y_min-65), cv2.FONT_HERSHEY_COMPLEX, 3, (218, 50, 127), 5)
                detected_class = classes[indexVal]
                text_output = f"{detected_class}"

            except:
                continue

    cv2.imshow('Imagem', img)
    key = cv2.waitKey(1) & 0xFF  # Ajuste o valor de espera para 10 milissegundos para uma reprodução mais rápida

    # Se a tecla 'q' for pressionada, interrompa o loop
    if key == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
