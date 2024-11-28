from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import load_image
mask1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask1.png")

output_height = 1024
output_width = 1024

processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask1], height=output_height, width=output_width)      

face_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png")

ip_images = [[face_image]]

masks = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]