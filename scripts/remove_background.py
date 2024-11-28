import os
from pathlib import Path
import sys
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

ROOT_DIR_PATH = Path(ROOT_DIR)
from paths import YOLO_MODEL_CACHE


from core.transform.yolo_person_transformator import YoloPersonTransformator
from core.utils.image import combine_images, combine_images_pil, load_image_from_path, remove_background, resize_image

imgage = load_image_from_path("./data/inputs/mairie-pacs.jpeg").convert("RGB")
resized_image, width, height = resize_image(imgage)
img = remove_background(resized_image)
img.save("./mairie-pacs-without-bg.png")

yolo = YoloPersonTransformator(yolo_cache=ROOT_DIR_PATH / YOLO_MODEL_CACHE)
        
mask =  yolo.isolate(img)
mask.save("./mairie-pacs-without-bg-mask.png")

result = load_image_from_path("./data/outputs/mairie-pacs1.png").convert("RGB")
result = remove_background(result)
result.save("./mairie-pacs-without-bg-result.png")

combined = combine_images_pil(resized_image, result, mask)
combined.save("./mairie-pacs-without-bg-combined.png")