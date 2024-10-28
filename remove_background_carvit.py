import torch
from carvekit.api.high import HiInterface
from PIL import Image

def transparent_to_black(image):
    # Obtenir les données de l'image (pixels)
    data = image.getdata()
    
    new_data = []
    
    # Parcourir les pixels de l'image
    for item in data:
        # item est sous la forme (R, G, B, A)
        if item[3] == 0:  # Si le pixel est transparent (Alpha = 0)
            new_data.append((0, 0, 0, 255))  # Remplacer par du noir opaque
        else:
            new_data.append(item)  # Sinon, garder le pixel tel quel
    
    # Mettre à jour les données de l'image avec les nouveaux pixels
    image.putdata(new_data)
    return image

def create_carvit_interface():
    interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)
    return interface

def remove_background(interface, image_path):
    images_without_background = interface([image_path])
    output_without_background = images_without_background[0]
    
    image_mask = transparent_to_black(output_without_background)
    return image_mask

#img = remove_background("./boulot-dams.jpeg")
#image_mask = transparent_to_black(img)