import cv2
import face_recognition

def extract_faces_opencv(image_path):
    # Charge l'image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Charge le modèle Haar Cascade pour la détection des visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Détecte les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Extrait chaque visage
    extracted_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        extracted_faces.append(face)
    
    return extracted_faces

# Exemple d'utilisation
faces = extract_faces_opencv(f"./data/inputs/mairie-pacs.jpeg")
print(f"Nombre de visages détectés: {len(faces)}")
for i, face in enumerate(faces):
    cv2.imwrite(f"./outputs/face_{i}.jpg", face)
