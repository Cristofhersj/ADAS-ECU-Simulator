# ADAS-ECU-Simulator
La compañía conocida como NI Costa Rica tenía el proyecto de crear un simulador de una ECU capaz de detectar objstáculos para vehículos en carretera en tiempo real. 

El proyecto consiste en generar una capa de percepción, la cual se encarga de detectar obstáculos comunes en el uso cotidiano de automóviles inteligentes. Se dio un enfoque en la investigación sobre distintas estrategias para lograr detectar y rastrear objetos en tiempo real. 
Tres algoritmos distintos fueron puestos a prueba para la aplicación: Yolov8, R-FCNN (Por medio de detectron2) y finalmente SSD por medio de YoloNAS, los tres algoritmos fueron puestos a prueba por medio del software "NI Monodrive" el cual permite generar escenarios virtuales para probar el funcionamiento correcto del algoritmo de detección y también la capacidad de enviar mensajes a otros sistemas del vehículo para lograr realizar acciones como frenado automático o bien reducción de velocidad de emergencia

En el archivo "Informe_Final_Completo_Solis_Cristofher" se detalla a fondo el proceso de diseño e implementación del proyecto. 

![image](https://github.com/Cristofhersj/ADAS-ECU-Simulator/assets/71050835/9fe092fa-1e83-42de-93b2-6e38ca950c4d)
