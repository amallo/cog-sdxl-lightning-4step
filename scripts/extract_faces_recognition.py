import face_recognition
from PIL import Image

def extract_faces(image_path):
    # Charge l'image
    image = face_recognition.load_image_file(image_path)
    
    # Localise les visages dans l'image
    face_locations = face_recognition.face_locations(image)
    
    # Extrait chaque visage et sauvegarde dans une liste
    faces = []
    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        faces.append(pil_image)
    
    return faces


def extract_head(image_path, padding=0.4):
    """
    Extrait une zone élargie autour du visage (tête complète) en ajoutant un padding.
    :param image_path: Chemin vers l'image.
    :param padding: Pourcentage d'agrandissement autour du visage.
    :return: Liste d'images PIL des zones élargies.
    """
    # Charge l'image
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    head_images = []
    for (top, right, bottom, left) in face_locations:
        # Calculer le padding
        height = bottom - top
        width = right - left
        padding_top = int(height * padding)
        padding_sides = int(width * padding)
        
        # Appliquer le padding en s'assurant de rester dans les limites de l'image
        top = max(0, top - padding_top)
        bottom = min(image.shape[0], bottom + padding_top)
        left = max(0, left - padding_sides)
        right = min(image.shape[1], right + padding_sides)
        
        # Extraire la zone élargie autour du visage
        head_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(head_image)
        head_images.append(pil_image)
    
    return head_images

# Exemple d'utilisation
faces = extract_faces(f"./data/inputs/mairie-pacs.jpeg")
faces_with_head = extract_head(f"./data/inputs/mairie-pacs.jpeg", 0.4)  
for i, face in enumerate(faces):
    face.save(f"./outputs/face_{i}.jpg")
for i, face in enumerate(faces_with_head):
    face.save(f"./outputs/head_{i}.jpg")
