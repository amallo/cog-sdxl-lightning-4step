import os
import sys
sys.path.append(os.path.abspath("/home/baba/work/cog-sdxl-lightning-4step"))

from core.transform.yolo_person_transformator import YoloPersonTransformator
from paths import SAM_MODEL_CACHE, YOLO_MODEL_CACHE
from scripts.generate_person_mask import ROOT_DIR


from core.utils.mask import generate_pytorch_face_mask
from PIL import Image
from core.utils.image import combine_images


resized_image  = Image.open(f"./data/inputs/mairie-pacs.jpeg")


faces_mask = generate_pytorch_face_mask(resized_image)
faces_mask.save(f"./faces_mask.png")
print("Created canny and depth maps")   


yolo_cache  = f"{ROOT_DIR}/{YOLO_MODEL_CACHE}"
sam_cache  = f"{ROOT_DIR}/{SAM_MODEL_CACHE}"

yolo = YoloPersonTransformator(yolo_cache=yolo_cache)
person_mask = yolo.isolate(resized_image)
person_mask.save(f"./person_mask.png")

combined_mask = combine_images(person_mask, faces_mask)
combined_mask.save(f"./combined_mask.png")