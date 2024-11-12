import cv2
import numpy as np

# Coordenadas das vagas (x, y, largura, altura)
vagas = [
    [1, 89, 108, 213], [115, 87, 152, 211], [289, 89, 138, 212],
    [439, 87, 135, 212], [591, 90, 132, 206], [738, 93, 139, 204],
    [881, 93, 138, 201], [1027, 94, 147, 202]
]

# Função para processar a imagem e verificar vagas
def processa_vaga(img, vagas):
    imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTh = cv2.adaptiveThreshold(
        imgCinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    imgBlur = cv2.medianBlur(imgTh, 5)
    kernel = np.ones((3, 3), np.int8)
    imgDil = cv2.dilate(imgBlur, kernel)

    qtVagasAbertas = 0
    for x, y, w, h in vagas:
        # Recorte da região da vaga
        recorte = imgDil[y:y + h, x:x + w]
        qtPxBranco = cv2.countNonZero(recorte)
        
        # Exibe a contagem de pixels brancos dentro da vaga
        cv2.putText(img, str(qtPxBranco), (x, y + h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Define a cor da vaga conforme se está ocupada ou não
        if qtPxBranco > 3000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            qtVagasAbertas += 1

    return img, imgDil, qtVagasAbertas

# Função principal
def main():
    video = cv2.VideoCapture('vaga.mp4')

    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    while True:
        check, img = video.read()
        
        # Verifica se o frame foi lido corretamente
        if not check:
            print("Não foi possível ler o frame.")
            break

        img, imgDil, qtVagasAbertas = processa_vaga(img, vagas)

        # Exibir a quantidade de vagas disponíveis na tela
        cv2.rectangle(img, (90, 0), (415, 60), (255, 0, 0), -1)
        cv2.putText(img, f'VAGAS DISPONIVEIS: {qtVagasAbertas}/{len(vagas)}',
                    (95, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

        # Mostra os frames processados
        cv2.imshow('Video', img)
        cv2.imshow('Detecção de Vagas', imgDil)

        # Sair do loop ao pressionar a tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Executa a função principal
if __name__ == "__main__":
    main()
