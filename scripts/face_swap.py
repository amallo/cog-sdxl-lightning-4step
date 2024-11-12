import cv2
import face_recognition
import numpy as np
from PIL import Image

def face_swap_multiple_sources(source_image_paths, target_image_path, output_path):
    # Charger l'image cible
    target_image = face_recognition.load_image_file(target_image_path)
    target_face_locations = face_recognition.face_locations(target_image, model="cnn")

    # Vérifier qu'il y a assez de visages dans la cible pour les visages sources
    if len(source_image_paths) > len(target_face_locations):
        print("Le nombre de visages dans la cible doit être supérieur ou égal au nombre de sources.")
        return

    # Convertir l'image cible en BGR pour OpenCV
    target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

    # Charger et traiter chaque visage source individuellement
    for i, source_image_path in enumerate(source_image_paths):
        # Charger l'image source pour ce visage
        source_image = face_recognition.load_image_file(source_image_path)
        source_face_location = face_recognition.face_locations(source_image, model="cnn")[0]
        source_landmarks = face_recognition.face_landmarks(source_image, [source_face_location])[0]

        # Convertir en BGR pour OpenCV
        source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)

        # Extraire la région du visage dans l'image source
        top, right, bottom, left = source_face_location
        source_face = source_image[top:bottom, left:right]

        # Sélectionner le visage cible correspondant
        target_face_location = target_face_locations[i]
        target_landmarks = face_recognition.face_landmarks(target_image, [target_face_location])[0]

        # Aligner les visages en utilisant les points de repère (landmarks)
        source_points = np.array([source_landmarks['left_eye'][0], source_landmarks['right_eye'][0], source_landmarks['nose_tip'][0]])
        target_points = np.array([target_landmarks['left_eye'][0], target_landmarks['right_eye'][0], target_landmarks['nose_tip'][0]])

        # Calcul de la matrice de transformation affine
        transformation_matrix = cv2.getAffineTransform(np.float32(source_points), np.float32(target_points))
        aligned_face = cv2.warpAffine(source_face, transformation_matrix, (target_image.shape[1], target_image.shape[0]))

        # Créer un masque pour la région du visage
        mask = np.zeros_like(target_image)
        target_top, target_right, target_bottom, target_left = target_face_location
        mask[target_top:target_bottom, target_left:target_right] = aligned_face

        # Appliquer le face swap pour chaque paire
        target_image[target_top:target_bottom, target_left:target_right] = cv2.seamlessClone(
            aligned_face,
            target_image,
            mask[target_top:target_bottom, target_left:target_right],
            (target_left + (target_right - target_left) // 2, target_top + (target_bottom - target_top) // 2),
            cv2.NORMAL_CLONE
        )

    # Convertir l'image de retour en RGB pour l'enregistrer
    result_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(result_image).save(output_path)


# Exemple d'utilisation
face_swap_multiple_sources([f"./outputs/face_1.jpg", f"./outputs/face_2.jpg"], f"./data/inputs/mairie-pacs-anciens.jpeg", "output_face_swap.jpg")
