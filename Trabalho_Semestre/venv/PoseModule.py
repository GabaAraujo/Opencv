import cv2 #tratamento de imagens
import mediapipe  as mp #lib para identificar movimentos e faces
import time
import math



class poseDetector(): #montar uma classe com todos os dados de cada objeto detectado, para modularização

    def __init__(self, mode=False,upBody = False, smooth = True,detectionCon=True,trackCon=0.5) -> None: #valores padroes ja definidos, podem ser alterados
    
        #  min_detection_confidence = 0.5 -> porcentagem de oque ele pode considerar uma pessoa ou nao para fazer o tracker
        # min_tracking_confidence = 0.5 -> he minimum confidence score for the pose tracking to be considered successful


        self.mode = mode # variavel da classe, atribuindo os valores ao objeto da classe detectados, cetando os valores definidos acima

        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose #captura a pose
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon) #pega o valor retornado e passa para o objeto



    def findPose(self, img, draw = True): #você quer fazer o display ou não.
       
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #CONVERSAO POIS OPENCV TRABALHA BGR E NAO RGB
        self.result = self.pose.process(imgRGB)
        if self.result.pose_landmarks: #se -> retorna resultado da marcacao
            if draw: 
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks,self.mpPose.POSE_CONNECTIONS) #escreve os dados,COM AS POSIÇÕES, DESENHA AS POSIÇÕES

        return img
   

    def getPosition(self,img,draw = True):
        self.lmList = []


        if self.result.pose_landmarks: #se -> retorna resultado da marcacao
          #MEDIAPIPE -> POSSUI INDICACOES DE OBJETOS 0->NOSE / 1 -> LEFT EYE / 5 -> RIGHT EYE
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape #retorno dos dados da imagem
               # print(id, lm) #numeração dos indicadores do mediapipe
                cx, cy = int(lm.x * w), int(lm.y * h) # MULTIPLICA AS MARCACOES COM O TAMANHO DA IMAGEM
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)  # DESENHA UM CIRCULO NOS VALORES -> X,Y RETORNADOS DAS POSIÇÕES DO CORPO


        return self.lmList
    

    def findAngle(self, img, p1,p2, p3, draw=True):

        #pega os pontos
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #calculo do angulo 
        #biceps
        #Retorna o ângulo cuja tangente é o quociente de dois números
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1-y2, x1-x2)) #comparando o angulo do centro aos 2 pontos do braco, 
        print(angle)


        #dedsenha o angulo
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0,0,255,1),5) #colore as linhas  blue,gree,red
            cv2.line(img, (x3, y3), (x2, y2), (0,0,255,1),5)

            cv2.circle(img, (x1, y1), 10, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255,0,0), 2)
            cv2.circle(img, (x2, y2), 10, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255,0,0), 2)
            cv2.circle(img, (x3, y3), 10, (0,255,0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255,0,0), 2)
            cv2.putText(img, str(int(angle)), (x2-50, y2+50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2) #printa o angulo

    

    #print(result.pose_landmarks) #mostra os resultados mapeados


   

        #MEDIAPIPE -> POSSUI INDICACOES DE OBJETOS 0->NOSE / 1 -> LEFT EYE / 5 -> RIGHT EYE

       # for id, lm in enumerate(result.pose_landmarks.landmark):
      #      h, w, c = img.shape #retorno dos dados da imagem
      #      print(id, lm) #numeração dos indicadores do mediapipe
      #      cx, cy = int(lm.x * w), int(lm.y * h) # MULTIPLICA AS MARCACOES COM O TAMANHO DA IMAGEM
     #       cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)  # DESENHA UM CIRCULO NOS VALORES -> X,Y RETORNADOS DAS POSIÇÕES DO CORPO


def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4') #abertura captura
    pTime = 0 #previus time => para definir o fps 1/TEMPOatual - Tempo previo
    detector = poseDetector()
    while True:
         success, img = cap.read() #retorna sucess se a captura der certo
         img = detector.findPose(img)
         lmList = detector.getPosition(img)
         if len(lmList) !=0:
            print(lmList[14]) #cada ponto representa um valor no desenho
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED) #faz uma marcacao no ponto desejado
         cTime = time.time() #pega o tempo atual
         fps = 1/(cTime-pTime)  #atual time - previus time
         pTime = cTime


         cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)


         cv2.imshow("Image", img) # mostra a imagem 
         cv2.waitKey(1) #delay de 1 millisegundo


if __name__ == "__main__": #executa a main
    main()