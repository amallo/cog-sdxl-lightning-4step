import os
import sys


sys.path.append(os.path.abspath("/home/baba/work/cog-sdxl-lightning-4step"))

from core.transform.yolo_person_transformator import YoloPersonTransformator
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from paths import SAM_MODEL_CACHE, YOLO_MODEL_CACHE
from scripts.generate_person_mask import ROOT_DIR
import torch
import mediapipe as mp
from core.utils.image import combine_images


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
    return mask


def generate_precise_face_mask(image: Image, include_parts=None):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            min_detection_confidence=0.5
        )

        # Convertir l'image PIL en format RGB
        if include_parts is None:
            include_parts = ['eyes', 'nose', 'mouth', 'face_oval']  # parties par défaut
            
        # Indices des points pour chaque partie du visage
        FACE_PARTS = {
            'eyes': [
                # Œil gauche
                [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
                # Œil droit
                [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            ],
            'eyebrows': [
                # Sourcil gauche
                [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
                # Sourcil droit
                [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            ],
            'nose': [
                # Nez
                [168, 193, 245, 188, 174, 217, 126, 142, 97, 98, 129, 49, 131, 134, 236, 239],
            ],
            'mouth': [
                # Bouche extérieure
                [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0],
                # Bouche intérieure
                [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13],
            ],
            'face_oval': [
                # Contour complet du visage (sens horaire)
                [10,   # Point de départ (menton)
                 338, 297, 332, 284,  # Joue gauche
                 251, 389, 356, 454, 323,  # Temple gauche
                 361, 288, 397, 365, 379,  # Front gauche
                 378, 400, 377, 152, 148,  # Front centre
                 176, 149, 150, 136, 172,  # Front droit
                 58, 132, 93, 234, 127,    # Temple droit
                 162, 21, 54, 103, 67,     # Joue droite
                 109, 10]                  # Retour au menton
            ]
        }

        # Convertir l'image PIL en format RGB
        image_rgb = image.convert('RGB')
        image_np = np.array(image_rgb)
        height, width = image_np.shape[:2]

        # Obtenir les résultats du face mesh
        results = face_mesh.process(image_np)
        
        # Créer un masque vide
        mask = Image.new('L', image.size, 255)
        draw = ImageDraw.Draw(mask)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convertir les coordonnées normalisées en pixels
                landmarks = [(int(point.x * width), int(point.y * height)) 
                           for point in face_landmarks.landmark]
                
                # Dessiner chaque partie sélectionnée
                for part in include_parts:
                    if part in FACE_PARTS:
                        for indices in FACE_PARTS[part]:
                            points = [landmarks[idx] for idx in indices]
                            draw.polygon(points, fill=0)
        return mask

image = Image.open(f"./data/inputs/mairie-pacs.jpeg")
face_mask = generate_precise_face_mask(image)
#mask = generate_face_mask(f"./data/inputs/mairie-pacs.jpeg")
face_mask.save("face_mask.png")


yolo_cache  = f"{ROOT_DIR}/{YOLO_MODEL_CACHE}"
sam_cache  = f"{ROOT_DIR}/{SAM_MODEL_CACHE}"

yolo = YoloPersonTransformator(yolo_cache=yolo_cache, sam_cache=sam_cache)
person_mask = yolo.isolate(image)
person_mask.save(f"./person_mask.png")


combined_mask = combine_images(person_mask, face_mask)
combined_mask.save(f"./combined_mask.png")