
import os
import sys

import cv2
import numpy as np
from rembg import remove
from transformers import CLIPSegProcessor, SegformerForSemanticSegmentation
import torch
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import torch
import torchvision.transforms as T
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F
import mediapipe as mp

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)

def pipe(image):

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return image, pred_seg

def remove_classes(image_path):
    original_image, pred_seg = pipe(image_path)

    # Convert to numpy arrays
    original_image_np = np.array(original_image)
    pred_seg_np = pred_seg.numpy()

    classes_to_remove = [4]

    combined_mask = np.isin(pred_seg_np, classes_to_remove)

    modified_image_np = np.zeros_like(original_image_np)

    modified_image_np[combined_mask] = [255]
    modified_image = Image.fromarray(modified_image_np.astype(np.uint8))
    return modified_image
def create_clothing_inpaint_mask(image_path):
    # 1. Charger l'image
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # 2. Charger le modèle et le processeur
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # 3. Prétraiter l'image
    inputs = processor(images=image, return_tensors="pt")

    # 4. Obtenir les prédictions
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_mask = torch.argmax(logits, dim=1)
    predicted_mask = predicted_mask.squeeze().cpu().numpy()

    # 5. Créer le masque pour les vêtements
    # Classes de vêtements (ajuster selon les besoins)
    clothing_classes = [4, 5, 6, 7, 8, 9, 10]  # Indices des classes de vêtements
    
    # Créer le masque binaire
    clothing_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
    for class_idx in clothing_classes:
        clothing_mask[predicted_mask == class_idx] = 255

    # 6. Inverser pour l'inpainting (noir = zone à modifier)
    inpainting_mask = 255 - clothing_mask

    return Image.fromarray(inpainting_mask)

def detect_people(image_path):
    """Détecte et découpe les personnes dans l'image"""
    # Utiliser YOLO ou MediaPipe pour la détection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    )
    
    image = cv2.imread(str(image_path))
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    people_boxes = []
    if results.pose_landmarks:
        h, w = image.shape[:2]
        for landmarks in [results.pose_landmarks]:  # Pour plusieurs personnes si disponible
            # Obtenir les coordonnées min/max
            x_coords = [lm.x * w for lm in landmarks.landmark]
            y_coords = [lm.y * h for lm in landmarks.landmark]
            
            # Ajouter une marge
            margin = 50
            x1 = max(0, int(min(x_coords)) - margin)
            y1 = max(0, int(min(y_coords)) - margin)
            x2 = min(w, int(max(x_coords)) + margin)
            y2 = min(h, int(max(y_coords)) + margin)
            
            people_boxes.append((x1, y1, x2, y2))
    
    return people_boxes, image

def virtual_try_on_multiple(image_path, cloth_path):
    """Applique l'essayage virtuel sur plusieurs personnes"""
    # 1. Détecter les personnes
    people_boxes, original_image = detect_people(image_path)
    
    # 2. Charger le vêtement
    cloth_image = Image.open(cloth_path)
    
    # 3. Traiter chaque personne
    results = []
    for i, box in enumerate(people_boxes):
        x1, y1, x2, y2 = box
        
        # Découper la personne
        person_img = original_image[y1:y2, x1:x2]
        person_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        
        # Sauvegarder temporairement
        temp_person_path = f"temp_person_{i}.jpg"
        person_img.save(temp_person_path)
        
        try:
            # Appliquer HR-VITON
            result = virtual_try_on(temp_person_path, cloth_path)
            results.append((box, result))
        except Exception as e:
            print(f"Erreur pour la personne {i}: {e}")
        
        # Nettoyer
        Path(temp_person_path).unlink()
    
    # 4. Combiner les résultats
    final_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    for box, result in results:
        x1, y1, x2, y2 = box
        # Redimensionner le résultat
        result = result.resize((x2-x1, y2-y1))
        # Coller dans l'image finale
        final_image.paste(result, (x1, y1))
    
    return final_image
     
image_path="./data/inputs/mairie-pacs.jpeg"
image = Image.open(image_path).convert("RGB")
modified_image=create_clothing_inpaint_mask(image)
modified_image.save("./mairie-pacs-clothing.jpeg")