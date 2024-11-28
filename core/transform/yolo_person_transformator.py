from paths import YOLO_MODEL_CACHE
from ultralytics import YOLO
from PIL import Image, ImageFilter, ImageDraw

import numpy as np

class YoloPersonTransformator:
    def __init__(self, yolo_cache: str):
        self.yolo = YOLO(yolo_cache)
       
    def isolate(self, image):
        # Charger l'image
        if image.mode == 'RGBA':
            # Créer un fond blanc
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            # Composer l'image sur le fond blanc
            image = Image.alpha_composite(background, image)
            # Convertir en RGB
            image = image.convert('RGB')
            
        # Détecter les personnes avec YOLO
        image_np = np.array(image)
        results = self.yolo(image_np)

        # Créer un masque vide
        mask = Image.new('L', image.size, 0)
       
        
        for result in results:
            # Vérifier si masks existe
            if hasattr(result, 'masks') and result.masks is not None:
                # Accéder aux masques
                for segment_mask in result.masks.data:  # Utiliser .data au lieu de .segments
                    # Convertir le masque en image PIL
                    mask_array = segment_mask.cpu().numpy()
                    # Redimensionner si nécessaire
                    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
                    mask_image = mask_image.resize(image.size)
                    
                    # Combiner avec le masque existant
                    mask = Image.fromarray(
                        np.maximum(
                            np.array(mask),
                            np.array(mask_image)
                        )
                    )
            
            # Dilater légèrement le masque pour inclure les bords des vêtements
            mask = mask.filter(ImageFilter.MaxFilter(3))
        
        return mask