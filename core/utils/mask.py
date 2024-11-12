from facenet_pytorch import MTCNN
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch




def generate_pytorch_face_mask(image):
    # Charger l'image
    #image = Image.open(image_path)
    width, height = image.size

    # Créer un masque noir de la même taille
    mask = Image.new("L", (width, height), 0)

    # Initialiser MTCNN pour la détection des visages
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Détecter les visages
    boxes, _ = mtcnn.detect(image)

    if boxes is not None:
        draw = ImageDraw.Draw(mask)
        for box in boxes:
            left, top, right, bottom = [int(coord) for coord in box]
            # Dessiner un rectangle blanc autour de chaque visage
            draw.rectangle([left, top, right, bottom], fill=255)

    # Sauvegarder le masque
    mask_np = np.array(mask)
    adjusted_mask = (255 - mask_np * 1.0).astype(np.uint8)
    inverted_mask = Image.fromarray(adjusted_mask)
    return inverted_mask