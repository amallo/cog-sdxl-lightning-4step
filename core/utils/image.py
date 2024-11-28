import shutil
import uuid
import cv2
from PIL import Image
import numpy as np
from rembg import remove
from diffusers.utils import load_image

def combine_images(canny_mask_path, face_mask_path):
    face_mask = cv2.imread(face_mask_path, cv2.IMREAD_GRAYSCALE)
    canny_image = cv2.imread(canny_mask_path, cv2.IMREAD_GRAYSCALE)
    # S'assurer que les deux masques sont de la même taille
    if canny_image.shape != face_mask.shape:
        face_mask = cv2.resize(face_mask, (canny_image.shape[1], canny_image.shape[0]))
    
    # Combiner les deux masques en utilisant une addition pondérée
    combined_mask = cv2.addWeighted(canny_image, 0.5, face_mask, 0.5, 0)
    
    return Image.fromarray(combined_mask)

def resize_to_allowed_dimensions(width, height):
        """
        Function re-used from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

def resize_image(image: Image):
        image_width, image_height = image.size
        print("Original width:"+str(image_width)+", height:"+str(image_height))
        new_width, new_height = resize_to_allowed_dimensions(image_width, image_height)
        print("new_width:"+str(new_width)+", new_height:"+str(new_height))
        image = image.resize((new_width, new_height))
        return image, new_width, new_height

def load_image_from_path(path: str):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    

def generate_unique_filename(filename: str, generation_id: str = uuid.uuid4()):
    return f"/tmp/{generation_id}-{filename}" 

def combine_images(mask1, mask2):
    # Convertir en arrays numpy
     # Convertir en arrays numpy
    mask1_array = np.array(mask1).astype(np.uint8)
    mask2_array = np.array(mask2).astype(np.uint8)
    
    # Copier les pixels noirs de mask2 dans mask1
    result = np.where(mask2_array == 0, 0, mask1_array)
    
    return Image.fromarray(result)

def remove_background(input_image):
    output_image = remove(input_image)
    return output_image


# merge two images with a mask
# mask1 is the image to keep
# transparent pixels in mask2 are replaced by the pixels of mask1
# other pixels in mask2 are copied as is
def merge_images(mask1, mask2):
    mask1_array = np.array(mask1).astype(np.uint8)
    mask2_array = np.array(mask2).astype(np.uint8)
    
    # Copier les pixels noirs de mask2 dans mask1
    result = np.where(mask2_array == 0, 0, mask1_array)
    
    return Image.fromarray(result)


def combine_images_pil(image1, image2, mask):
    # Convertir en RGB si nécessaire
    if image1.mode == 'RGBA':
        image1 = image1.convert('RGB')
    if image2.mode == 'RGBA':
        image2 = image2.convert('RGB')
    
    # Convertir le masque en mode 'L' si nécessaire
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # Combiner
    return Image.composite(image2, image1, mask)