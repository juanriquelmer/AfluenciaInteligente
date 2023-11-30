import os
import time
import cv2
from image_metrics import ImageMetrics


class CameraModule:
    """Clase para gestionar la cámara y procesar las imágenes capturadas."""
    def __init__(self, max_photos=100, camera_index=0, photo_directory="CameraModule_Log", capture_period=10, num_images_to_analyze=10):
        self.capturing = False  # Flag para saber si está capturando o no
        self.photo_count = 0  # Contador de fotos tomadas
        self.max_photos = max_photos  # Máximo de fotos que se pueden tomar
        self.photo_format = ".jpg"
        self.photo_directory = photo_directory
        self.camera_index = camera_index  # Índice de la cámara
        self.capture_period = capture_period  # Tiempo entre capturas
        self.num_images_to_analyze = num_images_to_analyze  # Número de imágenes a analizar

        # Crear directorio para las fotos si no existe
        if not os.path.exists(self.photo_directory):
            os.makedirs(self.photo_directory)

    def get_capturing(self):
        """Retorna el estado de captura."""
        return self.capturing

    # directory methods

    def _generate_photo_path(self, photo_number):
        """Genera y retorna la ruta completa de la foto basada en el número proporcionado."""
        photo_name = f"{photo_number:03d}{self.photo_format}"
        
        return os.path.join(self.photo_directory, photo_name)
    
    def _generate_last_photo_path(self):
        """Genera y retorna la ruta completa de la foto basada en el número proporcionado."""
        photo_name = f"{self.photo_count-1:03d}{self.photo_format}"
        
        return os.path.join(self.photo_directory, photo_name)
    
    def _generate_latest_image_paths(self):
        """Obtiene las rutas de las últimas imágenes basadas en self.num_images_to_analyze y teniendo en cuenta el ciclo de numeración."""
        image_numbers = [(self.photo_count - i - 1) % self.max_photos for i in range(self.num_images_to_analyze)] # Se obtienen los números de las últimas imágenes, se resta 1 para que no se incluya la imagen actual
        image_paths = [self._generate_photo_path(n) for n in image_numbers if os.path.exists(self._generate_photo_path(n))]
        
        return image_paths

    def save_photo(self, photo_number, frame):
        """Captura y guarda una foto."""
        photo_path = self._generate_photo_path(photo_number)
        cv2.imwrite(photo_path, frame)
        print(f"Photo {photo_path} saved.")

    def delete_photo(self, photo_number):
        """Elimina una foto basada en el número proporcionado."""
        photo_path = self._generate_photo_path(photo_number)
        if os.path.exists(photo_path):
            os.remove(photo_path)
            print(f"Foto {photo_path} eliminada.")
        else:
            #print("Error eliminando la foto.")
            #self._handle_error(error_type='delete_error')
            pass # Si no existe la foto no se hace nada
    # capture methods

    def initialize(self, numero_fotos_inicial = 10, initial_photo_period=1, max_retries=3, attemp_interval=0.25):
        """Inicializa el módulo de la cámara."""

        # eliminar las fotos del directorio
        for i in range(self.max_photos):
            self.delete_photo(i)

        # cargar las metricas de la imagen de referencia
        test_image_path = f"test.jpg"
        test_metrics = ImageMetrics.get_metrics(test_image_path)

        print(f" ------------------------- Métricas imagen de prueba -------------------------")
        print(f"  Métricas de la imagen de prueba: ")
        self.print_metrics(test_metrics)
        print(f"-----------------------------------------------------------------------------\n")

        # Se abre la cámara
        start_error = False # Flag para saber si hubo un error al inicializar el módulo
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self._handle_error('capture_error')
            start_error = True
            return start_error

        # Se toman las fotos iniciales
        for i in range(numero_fotos_inicial):
            # Capturar la foto
            ret, frame = self.cap.read()

            # Si no se pudo capturar la foto se manda el error altiro
            if not ret:
                start_error = True
            
            if self.compare_to_reference(frame, test_metrics): # Si la imagen es corrupta se puede intentar capturar de nuevo
                for attempt in range(max_retries):
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        self._handle_error(error_type='capture_error')
                        time.sleep(attemp_interval)
                    
                    if not self.compare_to_reference(frame, test_metrics):
                        break # La imagen es buena y podemos salir del bloque de los intentos

                    time.sleep(attemp_interval)

                    if attempt == max_retries - 1: # Si se llega al último intento y la imagen sigue siendo corrupta se manda el error  
                        start_error = True

            if start_error:
                # igualmente se guarda la foto para saber que pasó
                self.save_photo(i, frame)
                break 
            else:
                # En caso de que no haya errores se guarda la foto
                self.save_photo(i, frame)

                # Se actualiza el contador de fotos
                self.photo_count = (self.photo_count + 1) % self.max_photos

                # Se espera el tiempo de captura
                time.sleep(initial_photo_period)

        # Se cierra la cámara
        self.close_camera()

        return start_error

    def capture(self, max_retries=3, retry_delay=5, attemp_interval=1):
        """Comienza la captura de fotos en intervalos definidos por capture_period.
        
        Args:
            max_retries (int): Número máximo de intentos de captura.
            retry_delay (int): Delay entre intentos de captura.
        """

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self._handle_error('open_error')
            return

        while self.capturing:
            error_detected = False
            # Capturar la foto
            ret, frame = self.cap.read()
            
            # Si no se pudo capturar la foto se envía un error y se vuelven a intentar las capturas en un rato
            if not ret:
                self._handle_error(error_type='capture_error')
                error_detected = True

            # Verificar que la imagen esté bien
            if self.compare_image(frame): 
                # Se intena captura la foto de nuevo en un tiempo corto
                for attempt in range(max_retries):
                    print(f"Intento {attempt+1} de {max_retries}")
                    ret, frame = self.cap.read()
    
                    if not ret:
                        self._handle_error(error_type='capture_error')
                        time.sleep(attemp_interval)
                
                    if not self.compare_image(frame):
                        break # La imagen es buena y podemos salir
                    
                    time.sleep(attemp_interval)

                    if attempt == max_retries - 1:
                        error_detected = True
                        self._handle_error(error_type='corrupt_flag')
        
            if error_detected:
                time.sleep(retry_delay)
            else:
                # En caso de que no haya errores se guarda la foto
                self.save_photo(self.photo_count, frame)

                # Se actualiza el contador de fotos
                self.photo_count = (self.photo_count + 1) % self.max_photos
                
                # Se espera el tiempo de captura
                time.sleep(self.capture_period)

                error_detected = False

        # Se cierra la cámara
        self.close_camera()
    
    def close_camera(self):
        """Cierra la cámara."""
        print("Cerrando la cámara.")
        self.capturing = False
        self.cap.release()
        
    
    def compare_to_reference(self, test_frame, reference_image_metrics):
        """Evalúa una imagen con respecto a otra(o una imagen de referencia)"""
        bright_thr = 10
        var_lap_thr = 10
        hist_ent_thr = 1
        img_cont_thr = 10

        # Se obtienen las metricas de la imagen actual
        actual_image_metrics = ImageMetrics.get_metrics(test_frame, file_path=False)
        print(f" ------------------------- Métricas de la imagen actual -------------------------")
        self.print_metrics(actual_image_metrics)
        print(f"---------------------------------------------------------------------------------\n")

        corrupt_flag = False
        print(f" ------------------------- Evaluando imagen actual -------------------------")
        
        if actual_image_metrics['brightness'] <= reference_image_metrics['brightness'] + bright_thr:
            corrupt_flag = True
            print(f"    ALERTA: La imagen actual es mas oscura que la imagen de referencia.")
        
        if actual_image_metrics['variance_of_laplacian'] <= reference_image_metrics['variance_of_laplacian'] + var_lap_thr:
            corrupt_flag = True
            print(f"    ALERTA: La imagen actual es mas borrosa que la imagen de referencia.")
        
        if actual_image_metrics['histogram_entropy'] <= reference_image_metrics['histogram_entropy'] + hist_ent_thr:  
            corrupt_flag = True
            print(f"    ALERTA: La imagen actual tiene menos información que la imagen de referencia.")
    
        if actual_image_metrics['image_contrast'] <= reference_image_metrics['image_contrast'] + img_cont_thr:
            corrupt_flag = True
            print(f"    ALERTA: La imagen actual tiene menos contraste que la imagen de referencia.")


        print(f"\n")
        if corrupt_flag == False:
            print(f"    La imagen actual no tiene problemas.")
        print(f"-----------------------------------------------------------------------------\n")
        
        return corrupt_flag
    
    def print_metrics(self, metrics):
        """Imprime las metricas de una imagen"""
        print(f"Brillo: {metrics['brightness']}")
        print(f"Varianza del Laplaciano: {metrics['variance_of_laplacian']}")
        print(f"Entropía del histograma: {metrics['histogram_entropy']}")
        print(f"Contraste de la imagen: {metrics['image_contrast']}")

    def _evaluate_last_images_metrics(self):
        """Obtiene las metricas de el conjunto de las ultimas fotos"""

        last_images_paths = self._generate_latest_image_paths()
        print(f" ------------------------- Últimas imágenes -------------------------")
        print(f"  Últimas imágenes: {last_images_paths}")
        print(f"-----------------------------------------------------------------------\n")
        metrics_to_evaluate = [ImageMetrics.brightness_metric, ImageMetrics.variance_of_laplacian, ImageMetrics.histogram_entropy,  ImageMetrics.image_contrast]
        metrics_mapping = {
            ImageMetrics.brightness_metric: 'brightness',
            ImageMetrics.variance_of_laplacian: 'variance_of_laplacian',
            ImageMetrics.histogram_entropy: 'histogram_entropy',
            ImageMetrics.image_contrast: 'image_contrast'
        }
        results = {}
        for metric_func in metrics_to_evaluate:
            metric_name = metrics_mapping.get(metric_func, "Unknown Metric")
            results[metric_name] = ImageMetrics.analyze_image_set(last_images_paths, metric_func=metric_func)
        
        return results


    def compare_image(self, frame):
        """Evalúa las métricas de la última imagen capturada y las compara con las de las imágenes anteriores."""
        
        actual_image_metrics = ImageMetrics.get_metrics(frame, file_path=False)
        #print(f" ------------------------- Evaluando imagen actual -------------------------")
        #print(f"  Métricas de la última imagen:")
        #self.print_metrics(actual_image_metrics)
        #print(f"-----------------------------------------------------------------------------\n")

        last_images_metrics = self._evaluate_last_images_metrics()
              
        print(f" ------------------------- Evaluacion de la imagen comparada a las demas -------------------------")
        corrupt_flag = False
        for metric_name, metric_results in last_images_metrics.items():
            # Define el rango aceptable como el promedio más o menos dos desviaciones estándar
            limite_inferior = metric_results['mean'] - 5*metric_results['std']
            limite_superior = metric_results['mean'] + 5*metric_results['std']
            # Comprueba si la métrica está fuera de este rango
            if actual_image_metrics[metric_name] < limite_inferior or actual_image_metrics[metric_name] > limite_superior:
                corrupt_flag = True
                print(f"    ALERTA: La imagen actual está fuera del rango normal para {metric_name}.")

        print(f"----------------------------------------------------------------------------------------------------")

        return corrupt_flag


    def _handle_error(self, error_type=None):
        """Funcion que avisa si hay un error, por ahora solo imprime el error."""
        if error_type == 'corrupt_flag':
            print("Error: Imagen corrupta.")
        elif error_type == 'capture_error':
            print("-----------------------------\n")
            print("Error al capturar la foto.")
            print("-----------------------------\n")
        elif error_type == 'open_error':
            print("-----------------------------\n")
            print("Error al abrir la cámara.")
            print("-----------------------------\n")

        else:
            print("Unknown error encountered.")
