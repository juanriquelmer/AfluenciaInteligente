
import torch
import cv2
import supervision as sv

def process_image(image_path, model_weights_path, class_id=0, confidence_threshold=0.2):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', model_weights_path)
    
    # Read the image using OpenCV
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Unable to load the image from {image_path}. Check the file path and image format.")
        return None, 0
    
    # print(f"Image shape: {frame.shape}")
    
    # Perform object detection
    results = model(frame, size=1920)
    
    if results is None:
        print("Error: YOLOv5 did not return valid results.")
        return None, 0
    
    detections = sv.Detections.from_yolov5(results)
    
    # Filter detections by class_id and confidence
    # Ensure that the class_id and confidence_threshold are integers or floats
    detections = detections[(detections.class_id == int(class_id)) & (detections.confidence > float(confidence_threshold))]
    
    # Annotate the frame
    box_annotator = sv.BoxAnnotator(thickness=5, text_thickness=1, text_scale=0.5)
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    
    # Count the number of people
    num_people = len(detections)
    
    return annotated_frame, num_people

if __name__ == "__main__":
    IMAGE_PATH = "Tarea_imgs/IMG_1302.jpg"
    MODEL_WEIGHTS_PATH = 'yolov5x6'  # Specify the model weights you want to use
    annotated_frame, num_people = process_image(IMAGE_PATH, MODEL_WEIGHTS_PATH)
    
    if annotated_frame is not None:
        with sv.ImageSink(target_dir_path='Sink',image_name_pattern='processed_'+IMAGE_PATH.split('/')[1]) as sink:
                sink.save_image(image=annotated_frame)
        # Print the number of people in the picture
        print(f"Number of people in the picture: {num_people}")
