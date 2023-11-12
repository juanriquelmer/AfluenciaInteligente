from os import listdir
from cv2 import imread
from torch import int
from os.path import join, isfile
from ultralytics import YOLO

from ROI_extractor import image_divider # Codigo que extraen las ROI

def Runner(dir: str, modo: str, conf: int, weight: str):
    """
    Función main que cuenta la cantidad de personas detectadas en una imagen o
    imagenes de un directorio segun la ruta especificada en 'jpg_path'.

    Args:
        dir (str): Directorio de la imagen a analizar
        modo (str): Modo de lectura de la imagen, puede ser 'normal' o 'ROI'
        conf (int): Confianza del modelo
        weight (str): Peso del modelo YOLO utilizado para la detección

    Returns:
        int: Cantidad de personas detectadas en la imagen
    """
    image_data      = image_reader(dir, modo)
    person_detected = Counter(image_data, conf, weight)
    return person_detected

def image_reader(dir: str, modo: str = "normal"):
    """
    Función que lee una imagen desde una ruta y la retorna como un array de
    numpy.

    Args:
        path (str): Ruta de la imagen a leer
        modo (str): Modo de lectura de la imagen, puede ser 'normal' o 'ROI'

    Returns:
        np.array: Array de numpy con la imagen
    """
    if isfile(dir):
        print(f"Image: {dir}", end = "")
        if modo == "ROI":
            print(f" (Modo: {modo})")
            ROI_images_dir = image_divider(dir)
            imgs = read_from_folder(ROI_images_dir)
        else:
            print(f" (Modo: Normal)")
            imgs = [imread(dir)]
    else:
        print("Formato invalido! Solo se aceptan imagenes en formato .jpg")
    return imgs

def read_from_folder(dir: str):
    """
    Función que lee todas las imagenes de un directorio y las retorna como un
    array de numpy.

    Args:
        dir (str): directorio de las imagenes a leer

    Returns:
        lista: Entrega una lista con los arrays de numpy de las imagenes
    """
    images_data = []
    for jpg_file in listdir(dir):
        image_path = join(dir, jpg_file)
        images_data.append(imread(image_path))
    return images_data

def Counter(image_data: list, conf: int = 0.2, weight: str = "yolov8x.pt"):
    """
    Función que cuenta la cantidad de personas detectadas en una imagen o
    imagenes de un directorio segun la ruta especificada en 'jpg_path'.

    Args:
        image_data (list): Lista de imagenes a analizar leidas por opencv
        conf (float): Confianza del modelo
        model (str): Modelo YOLO especifico utilizado para la detección

    Returns:
        int: Numero de personas detectadas por el modelo
    """
    model = YOLO(weight)

    # Pesos base con 80 clases solo interesan personas:
    results = model.predict(image_data,
                            project     = "Predicciones",
                            save        = True,
                            save_txt    = True,
                            conf        = conf,
                            line_width  = 2,
                            # augment     = True,
                            classes     = 0
    )
    
    total = 0
    for i, result in enumerate(results):
        class_names         = result.names
        class_results       = result.boxes.cls.to(int).tolist()
        coordenate_results  = result.boxes.xywhn.tolist()
        output_dict         = dict()
        for c, bb in zip(class_results, coordenate_results):
            if class_names[c] not in output_dict:
                output_dict[class_names[c]] = []
            output_dict[class_names[c]].append(bb)

        # Cálculo de la cantidad de personas detectadas
        cantidad = 0
        for class_name in output_dict:
            cantidad += len(output_dict[class_name])
            print(f"image{i}: {cantidad} Personas (Model: {weight})")
        total += cantidad   # Suma de personas detectadas en todas imagen
    print(f"Total: {total} Personas (Model: {weight})")
    return total