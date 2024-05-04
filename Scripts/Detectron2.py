import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from roboflow import Roboflow
from datetime import datetime
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
setup_logger()

# Se definen parámetros que dependen del set de datos utilizado durante el entrenamiento

DATA_SET_NAME = "Car-finder-2"
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"

# TRAIN SET
TRAIN_DATA_SET_NAME = "Car-finder-2-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = "./Car-finder-2/train"
TRAIN_DATA_SET_ANN_FILE_PATH = "./Car-finder-2/train/_annotations.coco.json"

register_coco_instances(
    name=TRAIN_DATA_SET_NAME, 
    metadata={}, 
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH, 
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# TEST SET
TEST_DATA_SET_NAME = "Car-finder-2-test"
TEST_DATA_SET_IMAGES_DIR_PATH = "./Car-finder-2/test"
TEST_DATA_SET_ANN_FILE_PATH = "./Car-finder-2/test/_annotations.coco.json"

register_coco_instances(
    name=TEST_DATA_SET_NAME, 
    metadata={}, 
    json_file=TEST_DATA_SET_ANN_FILE_PATH, 
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)

# VALID SET
VALID_DATA_SET_NAME = "Car-finder-2-valid"
VALID_DATA_SET_IMAGES_DIR_PATH = "./Car-finder-2/valid"
VALID_DATA_SET_ANN_FILE_PATH = "./Car-finder-2/valid/_annotations.coco.json"

register_coco_instances(
    name=VALID_DATA_SET_NAME, 
    metadata={}, 
    json_file=VALID_DATA_SET_ANN_FILE_PATH, 
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  # @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

from IPython.display import clear_output
from detectron2.config import get_cfg
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME,)
cfg.DATASETS.TEST = (VALID_DATA_SET_NAME,)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 2000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05




cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
clear_output()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
cfg.DATASETS.TEST = (TEST_DATA_SET_NAME, )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get(TEST_DATA_SET_NAME)

from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()
# Se obtiene la entrada de vídeo o bien de imágenes
cap = cv2.VideoCapture("/home/datasets/ODT2/Escenario1.mp4")

frame_num = 0
# Object detection from Stable camera
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
if not os.path.exists('output_images2'):
    os.makedirs('output_images2')
while cap.isOpened():#True:
    ret, frame = cap.read()
    if not ret:  # End of video
        break
    height, width, _ = frame.shape
# Se lleva a cabo la detección y extracción de datos de la misma para poder utilizar el rastreo
    outputs = predictor(frame)
    bboxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    detections = []
    DATA = []
    for i, bbox in enumerate(bboxes):
            # Get coordinates of bounding box
            x1, y1, x2, y2 = bbox.astype(np.int32)
            class_name = classes[i]
            detections.append([x1, y1, x2, y2]) 
            #DATA.append()
            

    # Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x1, y1, x2, y2, id = box_id
        cv2.putText(frame, str(id), (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        DATA.append([x1, y1, x2, y2, class_name, id])
#Si se desea 
    #imS = cv2.resize(frame, (1024, 768)) 
    #Image(data=imS)
    #cv2.imshow("Frame", imS)visualizar los resultados en tiempo real se utilizan las líneas a continuación:
    output_path = f"output_images2/frame_{frame_num:04d}.jpg"
    cv2.imwrite(output_path, frame)
    frame_num+=1
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
clear_output()
print("Ready")
