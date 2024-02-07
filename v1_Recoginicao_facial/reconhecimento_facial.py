import cv2
import face_recognition as fr

imgUser = fr.load_image_file('img/julia_3.jpeg') #carrega a imagem
imgUser = cv2.cvtColor(imgUser,cv2.COLOR_BGR2RGB) # muta o padrao de cor
#imagem treinada


imgUserTest = fr.load_image_file('img/julia_1.jpeg') #carrega com o teste
imgUserTest = cv2.cvtColor(imgUserTest,cv2.COLOR_BGR2RGB) #muda o padrao de cor
#imagem a ser validada

imgUser = cv2.resize(imgUser, (720, 1080))
imgUserTest = cv2.resize(imgUserTest, (720, 1080))




faceLoc = fr.face_locations(imgUser)[0] #IDENTIFICA O ROSTO
cv2.rectangle(imgUser,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2) #BOTA NA CIRCUNFERENCIA

encodeElon = fr.face_encodings(imgUser)[0] # faz o processo de mapeamento
encodeElonTest = fr.face_encodings(imgUserTest)[0] #faz o processo de mapeamento

comparacao = fr.compare_faces([encodeElon],encodeElonTest)
distancia = fr.face_distance([encodeElon],encodeElonTest)

print(comparacao,distancia)
cv2.imshow('User',imgUser)
cv2.imshow('User Test',imgUserTest)
cv2.waitKey(0)