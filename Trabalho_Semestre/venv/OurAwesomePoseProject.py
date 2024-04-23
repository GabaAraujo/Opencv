import cv2
import time 
import  PoseModule as pm
import numpy as np

cap = cv2.VideoCapture('PoseVideos/VID.mp4') #abertura captura
#cap = cv2.VideoCapture('PoseVideos/4.jpg') 
pTime = 0 #previus time => para definir o fps 1/TEMPOatual - Tempo previo
detector = pm.poseDetector()

count = 0 #contador
dir = 0 

while True:
         success, img = cap.read() #retorna sucess se a captura der certo
         img = detector.findPose(img)
         lmList = detector.getPosition(img)
         if len(lmList) !=0:
            #print(lmList[14]) #cada ponto representa um valor no desenho
            #cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (0, 0, 255), cv2.FILLED) #faz uma marcacao no ponto desejado
            #braco esquerto
            angle = detector.findAngle(img, 12, 14, 16) #passando os locais onde vai ficar marcado
            #braco direito
            #detector.findAngle(img, 11, 13, 15) #passando os locais onde vai ficar marcado
            per = np.interp(angle,(80,150),(0,100)) # pega o angulo do exercicio -> de 80 a 150 - do video em questao

            bar = np.interp(angle,(80,150),(0,100)) # pega o angulo do exercicio -> de 80 a 150 - do video em questao
            #print(angle,per)








            if per == 100:
                  if dir == 0: #faz a contagem do movimento completo -> de 0 a 100% comparado pelo angulo 80=> 0 a 150 => 100%
                       count += 0.5
                       dir = 1

            if per == 0:
                  if dir == 1:
                        count +=0.5
                        dir = 0

            print(count)



           # cv2.rectangle(img, (1100,100),)


            #printa as repetuoes
            cv2.putText(img, str(int(count)), (600,120), cv2.FONT_HERSHEY_COMPLEX,5,(255,0,0),10)


         cTime = time.time() #pega o tempo atual
         fps = 1/(cTime-pTime)  #atual time - previus time
         pTime = cTime


         cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)


         cv2.imshow("Image", img) # mostra a imagem 
         cv2.waitKey(1) #delay de 1 millisegundo
