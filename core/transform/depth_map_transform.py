import numpy as np
import torch
from PIL import Image
from transformers import  DPTFeatureExtractor, DPTForDepthEstimation
from paths import DEPTH_ESTIMATION_CACHE, IMAGE_PROCESSOR_CACHE



class DepthMapTransform:
    def __init__(self, image_processor_cache: str, depth_estimation_cache: str):
        self.image_processor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=IMAGE_PROCESSOR_CACHE)
        self.depth_estimation = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=DEPTH_ESTIMATION_CACHE).to("cuda")
        

    def __call__(self, image: Image):
        image = self.image_processor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimation(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image