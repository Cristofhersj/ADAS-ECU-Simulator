import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import numpy as np
import os
import torch
import cv2
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)
from super_gradients.training import Trainer
import time
import csv
from PIL import Image
import requests
from io import BytesIO

from IPython.display import clear_output
import sys
from flask import Flask, render_template, Response
from queue import Queue
import threading 

sys.path.append(r'C:\Users\Beast\Desktop\Fiorella-monodrive-python-client\Algorithm2\examples\ODT2\ODT2')
from tracker import *
import os

# Se activa la gpu del computador en caso de estar disponible
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

# Se define la arquitectura del modelo y se importa el mismo

MODEL_ARCH = 'yolo_nas_s'
best_model = models.get(
    MODEL_ARCH,
    num_classes=4,
    checkpoint_path=r"C:\Users\Beast\Desktop\Fiorella-monodrive-python-client\Algorithm2\examples\average_model.pth"
).to(DEVICE)

colaStreaming = Queue()

# Create tracker object
tracker = EuclideanDistTracker()
frame_num = 0
filename = 'results.csv'
data=[]
# se define la función que se encarga de la detección y el rastreo de imágenes
def detect_track_ODT(frame):
    global frame_num
    global filename
    global data
    frame = np.ascontiguousarray(frame, dtype=np.uint8)


    
    # Object detection from Stable camera
    #object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    if not os.path.exists('output_images2'):
       os.makedirs('output_images2')
    
    height, width, _ = frame.shape

    ini=time.time()

    images_predictions = best_model.predict(frame)

    fini=time.time()
# Se extrae la información importante de la detección para realizar el rastreo
    for image_prediction in images_predictions:
      Objeto=[]
      class_names= image_prediction.class_names
      labels = image_prediction.prediction.labels
      confidence = image_prediction.prediction.confidence
      bboxes = image_prediction.prediction.bboxes_xyxy
    detections=[]
    for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
      print("prediction: ", i)
      print("label_id: ", label)
      print("label_name: ", class_names[int(label)])
      print("confidence: ", conf)
      print("bbox: ", bbox)
      print("--" * 10)
      x1 = int(bbox[0])
      y1=int(bbox[1])
      x2=int(bbox[2])
      y2=int(bbox[3])
      detections.append(bbox)

    # Se inicializa el rastreo de objetos
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
      x1, y1, x2, y2, idb = box_id
      cv2.putText(frame, str(idb), (int(x1), int(y1) - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
      Objeto.append([idb, class_names[int(label)], x1, y1, x2, y2])


# Se guardan los resultados y se envía la lista con el id, clase y posición de los objetos
    data.append ([frame_num, fini-ini])
    with open(filename, 'w', newline='') as file:
          writer = csv.writer(file)

      # Write the header row
          writer.writerow(['Iteration', 'Result'])
          
          # Write the data rows
          writer.writerows(data)
# Se se desea apreciar los resultados en tiempo real y/o guardar los fotogramas, se utilizan las siguientes líneas:
    #imS = cv2.resize(frame, (1024, 768)) 
    #Image(data=imS)
    #cv2.imshow("Frame" ,frame)
    #output_path = f"output_images2/frame_{frame_num:04d}.jpg" 

    cv2.imshow('Simulador',frame)
    cv2.waitKey(5)

    frame_num+=1
    #cv2.imwrite(output_path, frame)  
    return(Objeto)

#cap.release()
#cv2.destroyAllWindows()
#|clear_output()
print("Ready")

