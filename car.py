import cv2
import pytesseract
import time
import pandas as pd
import os

# Configuração do caminho para o executável Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Arquivo Excel onde os dados serão salvos
excel_file = 'dados_veiculos.xlsx'

# Cria um arquivo Excel se não existir
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=['Placa', 'Velocidade (km/h)', 'Hora', 'Data'])
    df.to_excel(excel_file, index=False)

# Função para salvar dados no arquivo Excel
def salvar_dados(placa, velocidade):
    df = pd.read_excel(excel_file)
    hora_atual = time.strftime('%H:%M:%S')
    data_atual = time.strftime('%Y-%m-%d')
    novo_dado = {'Placa': placa, 'Velocidade (km/h)': velocidade, 'Hora': hora_atual, 'Data': data_atual}
    df = pd.concat([df, pd.DataFrame([novo_dado])], ignore_index=True)
    df.to_excel(excel_file, index=False)

# Função para calcular a velocidade
def calcular_velocidade(distancia_percorrida_metros, tempo_segundos):
    if tempo_segundos == 0: 
        return 0
    velocidade_m_s = distancia_percorrida_metros / tempo_segundos  
    velocidade_kmh = velocidade_m_s * 3.6  
    return round(velocidade_kmh, 2)

# Caminho para o arquivo de vídeo
video_file_path = 'video.mp4'

# Captura o vídeo
video_capture = cv2.VideoCapture(video_file_path)

if not video_capture.isOpened():
    print(f"Erro ao abrir o arquivo de vídeo: {video_file_path}")
    exit()

# Carrega o classificador Haarcascade para detectar carros
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

if car_cascade.empty():
    print("Erro ao carregar o arquivo haarcascade_car.xml")
    exit()

# Variáveis para controlar o tempo e a distância
tempo_inicial = None
distancia_fixa = 5  # Distância fixa entre os dois pontos (em metros)
placas_detectadas = set()  # Para armazenar as placas já detectadas

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta carros no frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in cars:
        # Desenha um retângulo ao redor do carro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Captura a região onde a placa deve estar (ROI da placa)
        roi_placa = frame[y + int(h * 0.7):y + h, x:x + w]

        # Converte a ROI para escala de cinza
        placa_gray = cv2.cvtColor(roi_placa, cv2.COLOR_BGR2GRAY)

        # Aplica o threshold na imagem da placa para binarização
        _, placa_thresh = cv2.threshold(placa_gray, 150, 255, cv2.THRESH_BINARY)

        # Usa o Tesseract OCR para reconhecer o texto da placa
        placa_texto = pytesseract.image_to_string(placa_thresh, config='--psm 8').strip()

        # Verifica se o OCR reconheceu uma placa e se ela já não foi detectada
        if placa_texto and placa_texto not in placas_detectadas:
            # Adiciona a placa detectada ao conjunto de placas
            placas_detectadas.add(placa_texto)

            # Exibe a placa detectada no vídeo
            cv2.putText(frame, f'Placa: {placa_texto}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Calcula o tempo e a velocidade
            if tempo_inicial is None:
                tempo_inicial = time.time()  # Marca o tempo inicial
            else:
                tempo_final = time.time()
                tempo_decorrido = tempo_final - tempo_inicial

                if tempo_decorrido > 0.5:  # Para evitar medições de tempo muito curtas
                    velocidade = calcular_velocidade(distancia_fixa, tempo_decorrido)
                    cv2.putText(frame, f'Velocidade: {velocidade} km/h', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Salva os dados da placa e da velocidade
                    salvar_dados(placa_texto, velocidade)

                    # Reseta o tempo inicial
                    tempo_inicial = None

    # Exibe o vídeo com a detecção de placas e velocidades
    cv2.imshow('Video - Detecção de Placas e Velocidade', frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o vídeo e fecha as janelas
video_capture.release()
cv2.destroyAllWindows()
