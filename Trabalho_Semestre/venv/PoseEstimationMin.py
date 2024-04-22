import cv2 #tratamento de imagens
import mediapipe  as mp #lib para identificar movimentos e faces
import time

cap = cv2.VideoCapture('PoseVideos/1.mp4') #abertura captura
pTime = 0 #previus time => para definir o fps 1/TEMPOatual - Tempo previo

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose #captura a pose
pose = mpPose.Pose() #pega o valor retornado 


while True:
    success, img = cap.read() #retorna sucess se a captura der certo

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #CONVERSAO POIS OPENCV TRABALHA BGR E NAO RGB
    result = pose.process(imgRGB)
    print(result.pose_landmarks) #mostra os resultados mapeados


    if result.pose_landmarks: #se -> retorna resultado da marcacao
        mpDraw.draw_landmarks(img, result.pose_landmarks,mpPose.POSE_CONNECTIONS) #escreve os dados,COM AS POSIÇÕES, DESENHA AS POSIÇÕES

        #MEDIAPIPE -> POSSUI INDICACOES DE OBJETOS 0->NOSE / 1 -> LEFT EYE / 5 -> RIGHT EYE

        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = img.shape #retorno dos dados da imagem
            print(id, lm) #numeração dos indicadores do mediapipe
            cx, cy = int(lm.x * w), int(lm.y * h) # MULTIPLICA AS MARCACOES COM O TAMANHO DA IMAGEM
            cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)  # DESENHA UM CIRCULO NOS VALORES -> X,Y RETORNADOS DAS POSIÇÕES DO CORPO


    cTime = time.time() #pega o tempo atual
    fps = 1/(cTime-pTime)  #atual time - previus time
    pTime = cTime


    cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)


    cv2.imshow("Image", img) # mostra a imagem 
    cv2.waitKey(1) #delay de 1 millisegundo

   
