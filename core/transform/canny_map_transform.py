import cv2
import numpy as np
from PIL import Image



class CannyMapTransform:
    def __call__(self, image: Image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image