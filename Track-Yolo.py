import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import time
import csv
from IPython.display import clear_output
import sys
from flask import Flask, render_template, Response
from queue import Queue
import threading 

sys.path.append(r'C:\Users\Beast\Desktop\Fiorella-monodrive-python-client\Algorithm2\examples\ODT2\ODT2')
from tracker import *
import os

#Se importa el modelo de IA entrenado previamente
model= YOLO(r'C:\Users\Beast\Desktop\Fiorella-monodrive-python-client\Algorithm2\examples\ODT2\ODT2\best_aug.pt')


# Se genera el objeto de tipo rastreador
tracker = EuclideanDistTracker()

#Se definen variables importantes, el contador de fotogramas, la lista de datos que será la salida y el nombre del documento
#donde se almacenan los resultados de los datos (en este caso se guarda la velocidad de procesamiento en cada iteración)

frame_num = 0
filename = 'results.csv'
data=[]

#

def detect_track_ODT(frame):
    global frame_num
    global filename
    global data
    frame = np.ascontiguousarray(frame, dtype=np.uint8)

#Se genera una carpeta donde se guardan los fotogramas resultantes (se utiliza únicamente si se desea corroborar el rastreo)
    if not os.path.exists('output_images'):
       os.makedirs('output_images')
    
    height, width, _ = frame.shape

    ini=time.time()

    results = model.predict(frame) #En esta línea se hace el llamado a la función de Yolo para la detección de cada fotograma

    fini=time.time()
# Se extraen los datos relevantes para el sistema de control (Clase, coordenadas de los objetos)
    for result in results:
      Class=result.boxes.cls
      Clase=Class.cpu().numpy()
      Coordenadas=result.boxes.xyxy
      Tamaño=Coordenadas.cpu().numpy()
      Objeto=[]
      if len(Clase)>=1:
        for i in range(len(Clase)):
          Sublist=[]
          Claseint= int(Clase[i])
          Sublist.append(Claseint)
          Sublist.append(Tamaño[i])
          Objeto.append(Sublist)
    detections = []
    for i in range (len(Tamaño)):
      x1,y1,x2,y2 = Tamaño[i]
      x1= int(x1)
      x2= int(x2)
      y1 = int(y1)
      y2 = int(y2)
      detections.append([x1, y1, x2, y2]) 



    # Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x1, y1, x2, y2, idb = box_id
        cv2.putText(frame, str(idb), (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #DATA.append([idb, Claseint, x1, y1, x2, y2])
        DATA.append([idb, Claseint, x1, y1, x2, y2])

    print("DATOS QUE SE VAN A ENVIAR:", DATA)
    data.append ([frame_num, fini-ini])
    with open(filename, 'w', newline='') as file:
          writer = csv.writer(file)

      # Write the header row
          writer.writerow(['Iteration', 'Result'])
          
          # Write the data rows
          writer.writerows(data)
# Si se desea visualizar en tiempo real y/o guardar los fotogramas resultantes, se pueden utilizar las lineas a continuación
    #imS = cv2.resize(frame, (1024, 768)) 
    #Image(data=imS)
    #cv2.imshow("Frame" ,frame)
    #output_path = f"output_images2/frame_{frame_num:04d}.jpg" 

    cv2.imshow('Simulador',frame)
    cv2.waitKey(5)

    frame_num+=1
    #cv2.imwrite(output_path, frame)  
    return(DATA)

#cap.release()
#cv2.destroyAllWindows()
#|clear_output()
print("Ready")

