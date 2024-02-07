import cv2
import face_recognition
import pickle #serve para converter um arquivo em um conjunto de bytes e serializar ela
import os


folderPath = 'Images'

pathList = os.listdir(folderPath) #lista todas as imagens do folder, com o nome de cada arquivo
#print(pathList)

imgList = [] #pegar lista de imagens
studentIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])  # registra os valores no id dos estudantes


    #print(path)
    #print(os.path.splitext(path)[0]) #retornar um array com 2 valores, 0 -> nome do arquivo/1 -> extensão do arquivo


print(studentIds) #lista a quantidade de Ids registrados baseado no nome da base

def findEncodings(imagesList): #faz a conversao de todas as imagens -> funcao
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] #faz a analise de reconhecimento facial das imagens
        encodeList.append(encode) #faz a serializacao

    return encodeList

print("Encoding Started . . .")
encodeListKnow = findEncodings(imgList)
encodeListKnowWithIds = [encodeListKnow, studentIds] #encode com o dados serializado da deteccao e do nome da imagem
print(encodeListKnow)
print("Enconding Complete")


file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnowWithIds, file)
file.close