import os
import pickle
import cv2

cap = cv2.VideoCapture('video/video3.mp4')
cap.set(3, 640)
cap.set(4, 480)

#imgBackground = cv2.imread('Resources/background.png')
imgBackground = cv2.imread('Resources/background.png')

folderModePath = 'Resources/Modes/' #localizacao do folder do menu
modePathList = os.listdir(folderModePath) #lista todas as imagens do folder, com o nome de cada arquivo
imgModeList = [] #pegar lista de imagens


for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

print(modePathList) #lista os nomes
print(len(imgModeList)) #lista a quantidade

#carregar o encodi file serializado.
print("Encode File")
file = open('EncodeFile.p','rb') #leitura e gravacao
encodeListKnowWithIds = pickle.load(file)
file.close()
encodeListKnow, studentIds = encodeListKnowWithIds
print("Encode File Loaded")

#print(studentIds)



while True:
    success, img = cap.read()
    #imgBackground[162:162 + 480, 55:55 + 640] = img #colocar camera depois
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[2]


    #cv2.imshow('Video', img)

    cv2.imshow("Face Attendence", imgBackground)
    cv2.waitKey(1)

#for path in folderModePath:
