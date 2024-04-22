import cv2
import time 
import  PoseModule as pm

#cap = cv2.VideoCapture('PoseVideos/1.mp4') #abertura captura
cap = cv2.VideoCapture('PoseVideos/4.jpg') 
pTime = 0 #previus time => para definir o fps 1/TEMPOatual - Tempo previo
detector = pm.poseDetector()



while True:
         success, img = cap.read() #retorna sucess se a captura der certo
         img = detector.findPose(img)
         lmList = detector.getPosition(img)
         if len(lmList) !=0:
            print(lmList[14]) #cada ponto representa um valor no desenho
            #cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (0, 0, 255), cv2.FILLED) #faz uma marcacao no ponto desejado
            detector.findAngle(img, 12, 14, 16) #passando os locais onde vai ficar marcado

         cTime = time.time() #pega o tempo atual
         fps = 1/(cTime-pTime)  #atual time - previus time
         pTime = cTime


         cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)


         cv2.imshow("Image", img) # mostra a imagem 
         cv2.waitKey(0) #delay de 1 millisegundo
