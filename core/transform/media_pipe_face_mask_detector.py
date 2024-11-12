from PIL import Image, ImageDraw
import mediapipe as mp
import numpy as np

class MediaPipeFaceMaskDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            min_detection_confidence=0.5
        )
    def __call__(self, image: Image, include_parts=None):
        print("Loading face mesh")
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
        results = self.face_mesh.process(image_np)
        
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
