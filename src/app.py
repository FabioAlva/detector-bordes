import numpy as np
import matplotlib.pyplot as plt

def leer_y_convertir_gris(path_imagen):
    """Lee la imagen y la convierte a escala de grises."""
    imagen = plt.imread(path_imagen)
    # Si la imagen tiene canal alpha (PNG), nos quedamos con RGB
    if imagen.ndim == 3 and imagen.shape[2] >= 3:
        imagen = imagen[:, :, :3]
    filter_gray = np.array([0.299, 0.587, 0.114])
    return np.dot(imagen, filter_gray)

def detectar_bordes_sobel(imagen_gray):
    """Aplica el detector de bordes de Sobel utilizando convolución espacial."""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    alto, ancho = imagen_gray.shape
    imagen_bordes = np.zeros((alto - 2, ancho - 2))

    for i in range(alto - 2):
        for j in range(ancho - 2):
            submatriz = imagen_gray[i:i+3, j:j+3]
            
            # Convolución en X e Y
            val_x = np.sum(submatriz * sobel_x)
            val_y = np.sum(submatriz * sobel_y)
            
            # Magnitud del gradiente
            imagen_bordes[i, j] = np.sqrt(val_x**2 + val_y**2)
            
    return imagen_bordes

def mostrar_resultado(imagen, titulo="Bordes Detectados"):
    """Muestra la imagen resultante con matplotlib."""
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    try:
        imagen_gris = leer_y_convertir_gris('./images_2.jpg')
        bordes = detectar_bordes_sobel(imagen_gris)
        mostrar_resultado(bordes)
    except FileNotFoundError:
        print("Error: No se encontró la imagen './images_2.jpg'. Verificá la ruta.")