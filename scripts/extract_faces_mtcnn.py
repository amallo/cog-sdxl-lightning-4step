from facenet_pytorch import MTCNN
from PIL import Image

import numpy as np
import cv2

def extract_faces_with_mtcnn(image_path):
    mtcnn = MTCNN(keep_all=True)
    image = Image.open(image_path)
    
    # Détecte les visages avec MTCNN
    boxes, _ = mtcnn.detect(image)
    
    faces = []
    for i, box in enumerate(boxes):
        left, top, right, bottom = [int(coord) for coord in box]
        face = image.crop((left, top, right, bottom))
        faces.append(face)
    
    return faces


def extract_and_interpolate_faces(image_path):
    mtcnn = MTCNN(keep_all=True)
    image = Image.open(image_path)
    
    # Détection des visages
    boxes, _ = mtcnn.detect(image)
    
    faces = []
    for i, box in enumerate(boxes):
        left, top, right, bottom = [int(coord) for coord in box]
        
        # Extraire le visage sans padding
        face = image.crop((left, top, right, bottom))
        
        # Convertir l'extraction en format numpy pour l'interpolation si besoin
        face_np = np.array(face)
        
        # Création d'une bordure pour le visage
        face_with_border = cv2.copyMakeBorder(
            face_np,
            20, 20, 20, 20,  # 20 pixels de bordure autour du visage
            cv2.BORDER_REFLECT
        )
        
        # Ajout de l'image avec bordure à la liste des visages
        faces.append(Image.fromarray(face_with_border))
    
    return faces

# Exemple d'utilisation
faces = extract_and_interpolate_faces("./data/inputs/mairie-pacs.jpeg")
print(f"Nombre de visages détectés: {len(faces)}")
for i, face in enumerate(faces):
    face.save(f"./outputs/face_mtcnn_{i}.jpg")
