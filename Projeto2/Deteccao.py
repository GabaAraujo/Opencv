import cv2
import json
import base64
import requests
from google.cloud import vision



img = cv2.imread('imagens/004.jpg') #carrega a imagem em uma variavel
cv2.imshow('img', img) #abre a iamgem carregada

cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte o RGB para escala de cinza, para melhor detecção
#cv2 .imshow('cinza', cinza)#abre a iamgem carregada convertida

#transformar a imagem linearizada -> para ter 2 cores  - lim min:branco / lim max: preto

#tem de retornar um valor nessa funcao, pode ser ate vazio
_,binaria = cv2.threshold(cinza, 90, 255, cv2.THRESH_BINARY) #funcao de binarizacao -> img_a_ser_carregada,lim_max=preto,lim_max=branco,funcao_de_linariazacao

#cv2.imshow('binaria', binaria)

#desfoque da imagem para melhrar as formas geometricas
#desfoque = cv2.GaussianBlur(binaria, (5,5), 0) #desfoque, sendo o menor valor possivel
#cv2.imshow('desfoque', desfoque)


#procurar contorno da imagem
contornos, hier = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #identica os contornos. RETORNANDO 3 PARAMETROS -> contorno dentro de contornos
#retorna 2 parametros ->os contornos / hierarquia das imagens

#retornar um array de contornos

#print(contornos)

#desenha contornos na imagem

#cv2.drawContours(img, contornos, -1, (0,255,0),3) #sera printado na imagem original, sera colocado os contornos,cada forma descoberta tem um id-> como queremos todas e -1, sera plotado a cor verde,espessura
#cv2.imshow('img com contornos', img)


for c in contornos:
    perimetro = cv2.arcLength(c,True)#pega apenas contornos fechados
    if perimetro > 1450 and perimetro < 1550:
        aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True) #faz uma aproximacao para a forma geometrica dela

        if len(aprox) == 4: #eu quero um quandrado ou retangulo, não quero outras formas geometricas
            (x,y, alt,larg) = cv2.boundingRect(c) # possui/x incial/y final / altura/ largura -> transformar o formato em retangulo
            #pois mesmo com 4 lados e fechado pode ser que não seja um retangulo,então e preciso filtrar

            cv2.rectangle(img, (x,y), (x+alt, y+larg), (0,255,0),3) #printar na imagem o retangulo





cv2.imshow('img pintada', img)
cv2.waitKey(0) #espera uma key para fechar
cv2.destroyAllWindows() #fecha as janelas






def GoogleApi():

    with open("imagens/000.jpg", "rb") as img_file:
        my_base64 = base64.b64encode(img_file.read())

    url = "https://vision.googleapis.com/v1/images:annotate?key=AIzaSyBLGgG9-0E_igEcbR91nyb6o4X0c8mX1w4"
    data = {
        'requests': [
            {
                'image': {
                    'content': my_base64.decode('utf-8')
                },
                'features': [
                    {
                        'type': 'TEXT_DETECTION'
                    }
                ]
            }
        ]
    }

    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())


if  __name__  == "__main__":
    GoogleApi()
