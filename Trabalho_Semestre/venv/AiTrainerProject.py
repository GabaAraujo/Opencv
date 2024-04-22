import cv2
import time 
import  PoseModule as pm


cap = cv2.VideoCapture('PoseVideos/1.mp4') #abertura captura
pTime = 0 #previus time => para definir o fps 1/TEMPOatual - Tempo previo
detector = pm.poseDetector()



while True:

        
        img2 = cv2.imread('PoseVideos/4.jpg')
        img2 = detector.findPose(img2)
        

        cv2.imshow("Image", img2) # mostra a imagem 
        cv2.waitKey(1) #delay de 1 millisegundo

