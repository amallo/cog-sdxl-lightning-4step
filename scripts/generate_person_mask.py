import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from core.transform.yolo_person_transformator import YoloPersonTransformator

from paths import SAM_MODEL_CACHE, YOLO_MODEL_CACHE

from PIL import Image


resized_image  = Image.open(f"{ROOT_DIR}/data/inputs/mairie-pacs.jpeg")

yolo_cache  = f"{ROOT_DIR}/{YOLO_MODEL_CACHE}"
sam_cache  = f"{ROOT_DIR}/{SAM_MODEL_CACHE}"

yolo = YoloPersonTransformator(yolo_cache=yolo_cache, sam_cache=sam_cache)
faces_mask = yolo.isolate(resized_image)
faces_mask.save(f"./person_mask.png")