import cv2
import time 
import PoseModule as pm
import numpy as np

cap = cv2.VideoCapture('PoseVideos/VID.mp4')  # Abertura da captura
pTime = 0  # Tempo anterior para cálculo de FPS
detector = pm.poseDetector()

count = 0  # Contador de repetições
dir = 0  # Direção do movimento (0 para baixo, 1 para cima)

while True:
    success, img = cap.read()  # Leitura da imagem
    img = detector.findPose(img)
    lmList = detector.getPosition(img, draw=False)  # Alteração para evitar desenho automático dos pontos

    if len(lmList) != 0:
        angle = detector.findAngle(img, 12, 14, 16)  # Ângulo do braço esquerdo
        per = np.interp(angle, (80, 150), (0, 100))  # Porcentagem do movimento

        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0

        # Barra de progresso no canto inferior direito
        bar_start = int(np.interp(per, (0, 100), (720, 200)))  # Altura inicial variável conforme percentual
        cv2.rectangle(img, (1240, 720), (550, bar_start), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (550, 150), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)

        # Exibição do número de repetições
        cv2.putText(img, f'Reps: {int(count)}', (550, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

    # Cálculo de FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Mostra a imagem
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

cap.release()
cv2.destroyAllWindows()
