import os
import pandas as pd
import xml.etree.ElementTree as xet

# Definindo o caminho da pasta onde estão os arquivos XML
pasta_xml = 'dataset/Dataset Previus/annotations'

# Função para fazer o parsing de cada arquivo XML e extrair as informações necessárias
def parsing(path):
    parser = xet.parse(path).getroot()
    
    # Caminho para o arquivo de imagem
    name = parser.find('filename').text
    filename = f'dataset/Dataset Previus/images/{name}'  # Ajuste conforme necessário
    
    # Extraindo largura e altura da imagem
    parser_size = parser.find('size')
    width = int(parser_size.find('width').text)
    height = int(parser_size.find('height').text)
    
    # Extraindo as coordenadas do objeto 'licence'
    bndbox = parser.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    
    return path, xmin, xmax, ymin, ymax, filename, width, height

# Criando uma lista para armazenar as informações extraídas
data = []

# Iterando por todos os arquivos XML na pasta
for arquivo in os.listdir(pasta_xml):
    if arquivo.endswith('.xml'):
        caminho_arquivo = os.path.join(pasta_xml, arquivo)
        data.append(parsing(caminho_arquivo))

# Convertendo a lista de dados em um DataFrame com as colunas desejadas
df = pd.DataFrame(data, columns=['filepath', 'xmin', 'xmax', 'ymin', 'ymax', 'filename', 'width', 'height'])

# Calculando center_x, center_y, largura e altura da bounding box normalizados
df['center_x'] = (df['xmax'] + df['xmin']) / (2 * df['width'])
df['center_y'] = (df['ymax'] + df['ymin']) / (2 * df['height'])

df['bb_width'] = (df['xmax'] - df['xmin']) / df['width']
df['bb_height'] = (df['ymax'] - df['ymin']) / df['height']

# Salvando o DataFrame como um arquivo CSV
df.to_csv('dataset/Dataset Previus/labels.csv', index=False)


# Dividindo o DataFrame em treino e teste
df_train = df.iloc[:200]  # 200 primeiros registros para treino
df_test = df.iloc[200:]   # Restante para teste



train_folder = 'yolov7/data_images/train'
test_folder = 'yolov7/data_images/test'


from shutil import copy

# Função para copiar imagens e gerar os arquivos de labels .txt
def copy_images_and_generate_labels(df, folder):
    values = df[['filename', 'center_x', 'center_y', 'bb_width', 'bb_height']].values
    for fname, x, y, w, h in values:
        image_name = os.path.split(fname)[-1]
        txt_name = os.path.splitext(image_name)[0]
        
        # Caminhos de destino
        dst_image_path = os.path.join(folder, image_name)
        dst_label_file = os.path.join(folder, txt_name + '.txt')
        
        # Copiando a imagem
        copy(fname, dst_image_path)
        
        # Gerando o arquivo .txt com as informações
        label_txt = f'0 {x} {y} {w} {h}'
        with open(dst_label_file, mode='w') as f:
            f.write(label_txt)
            f.close()

# Copiar imagens e gerar labels para treino e teste
copy_images_and_generate_labels(df_train, train_folder)
copy_images_and_generate_labels(df_test, test_folder)


yaml_content = """
train: data_images/train
val: data_images/test
nc: 1
names: ['license_plate']
"""

# Caminho do arquivo data.yaml
yaml_path = './yolov7/data.yaml'

# Salvando o arquivo data.yaml
with open(yaml_path, 'w') as f:
    f.write(yaml_content)



# Exibindo as primeiras linhas para ver o resultado
df.head()
