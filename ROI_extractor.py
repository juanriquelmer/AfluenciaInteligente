from os import makedirs
from cv2 import imread, imwrite
from os.path import join, basename

def image_divider(imagen_path: str):
    """_summary_

    Args:
        imagen_path (str): Ruta de la imagen a extraer las Regiones De Interes

    Returns:
        str: Directorio donde se guardaran las ROI
    """
    image_name  = basename(imagen_path).split('.')[0]
    output_path = f"ROI_Images/{image_name}"
    makedirs(output_path, exist_ok = True)
    extraer_4zonas(imagen_path, output_path)
    extraer_cruz(imagen_path, output_path)
    return output_path

def extraer_4zonas(imagen_path: str, output_path: str):
    """
    Divide la imagen en 4 iguales, las zonas de interes, guardando cada zona en
    un archivo separado.

    Args:
        imagen_path (str): Ruta de la imagen a separar en Regiones De Interes

    Returns:
        str: Ruta donde se guardan las imagenes resultantes (ROI)
    """
    # Cargar la imagen
    imagen = imread(imagen_path)

    # Obtener dimensiones de la imagen
    alto, ancho = imagen.shape[:2]
    mitad_alto, mitad_ancho = alto//2, ancho//2
    # Dividir la imagen en cuatro partes
    partes = [imagen[:mitad_alto, :mitad_ancho], imagen[:mitad_alto, mitad_ancho:],
              imagen[mitad_alto:, :mitad_ancho], imagen[mitad_alto:, mitad_ancho:]]
    
    # Guardar las partes divididas
    for i, parte in enumerate(partes):
        imwrite(join(output_path, f"ROI_{i}.jpg"), parte)

def extraer_cruz(imagen_path: str, output_path: str):
    """
    Extrae las cruces de la imagen original y las guarda en archivos separados.

    Args:
        imagen_path (str): Ruta de la imagen a separar en Regiones De Interes
        output_path (str): Directorio donde se guardaran las ROI
    """
    # Cargar la imagen
    imagen = imread(imagen_path)

    # Obtener dimensiones de la imagen
    alto, ancho = imagen.shape[:2]

    # Especificar las coordenadas y dimensiones de la cruz
    margen_h = 200  # Margen desde el centro
    margen_v = 50  # Margen desde el centro
    mitad_alto, mitad_ancho = alto//2, ancho//2
    inicio_horizontal   = mitad_ancho - margen_h
    fin_horizontal      = mitad_ancho + margen_h
    inicio_vertical     = mitad_alto - margen_v
    fin_vertical        = mitad_alto + margen_v

    # Extraer la cruz de la imagen original
    cruz_vertical_sup   = imagen[mitad_alto:, inicio_horizontal:fin_horizontal]
    cruz_vertical_inf   = imagen[:mitad_alto, inicio_horizontal:fin_horizontal]
    cruz_horizontal_izq = imagen[inicio_vertical:fin_vertical, :mitad_ancho]
    cruz_horizontal_der = imagen[inicio_vertical:fin_vertical, mitad_ancho:]

    # Guardar las cruces extraídas
    imwrite(join(output_path, f"ROI_CV_Sup.jpg"), cruz_vertical_sup)
    imwrite(join(output_path, f"ROI_CV_Inf.jpg"), cruz_vertical_inf)
    imwrite(join(output_path, f"ROI_CH_Izq.jpg"), cruz_horizontal_izq)
    imwrite(join(output_path, f"ROI_CH_Der.jpg"), cruz_horizontal_der)