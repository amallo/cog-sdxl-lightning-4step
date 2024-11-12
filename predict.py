# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import uuid
from cog import BasePredictor, Input, Path
import os
import time
from core.cache.download_weights import DownloadWeights
from core.transform.yolo_person_transformator import YoloPersonTransformator
from paths import BASE_CACHE, CONTROLNET_CANNY_PATH, CONTROLNET_DEPTH_PATH, DEPTH_ESTIMATION_CACHE, FEATURE_EXTRACTOR, IMAGE_PROCESSOR_CACHE, LORA_CACHE_CUSTOM, REFINER_MODEL_CACHE, SAFETY_CACHE, YOLO_MODEL_CACHE
import torch
from typing import List
from pathlib import Path as LocalPath

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers import ControlNetModel, DiffusionPipeline, DPMSolverMultistepScheduler
from core.pipelines.schedulers import SCHEDULERS
from core.pipelines.sdxl_background_pipeline import SdxlBackgroundPipeline
from core.pipelines.sdxl_outfit_pipeline    import SdxlOutfitPipeline
from core.transform.depth_map_transform import DepthMapTransform
from core.transform.canny_map_transform import CannyMapTransform
from core.transform.media_pipe_face_mask_detector import MediaPipeFaceMaskDetector
from transformers import CLIPImageProcessor


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = LocalPath(ROOT_DIR)

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


default_landscape_prompt="""
An abandoned area left in ruins since the civil war, overgrown vegetation, buildings riddled with bullet holes. 
The ground littered with abandoned objects and debris. 
wrecked vehicules. 
Cracks and fractured surfaces. The atmosphere is eerie and dark, a post-apocalyptic ambiance. 
Graffitis and tags cover the walls, crumbling structures. 
The scene is captured with high photographic realism. 
abandoned objects, scattered debris, graffiti, overwhelming desolation. 
"""

default_outfit_prompt="""
People wears post apocalyptic outfit: cargo pants, bulletproof vests with multiple pockets hold ammo, knives in homemade sheaths, and handguns tucked into belts. their faces covered with scars. They carry makeshift weapons, knives in homemade sheaths, and handguns tucked.
"""

default_camera_prompt="""
High details, Photorealistic, FUJIFILM X-T5,35mm,focal length 53mm,ƒ/1.4 aperture, 1/320s shutter speed, ISO 125, and +2/3 exposure compensation
"""
class Predictor(BasePredictor):
    
    def setup(self) -> None:
        download_weights = DownloadWeights(ROOT_PATH)
        download_weights()
        feature_extractor = CLIPImageProcessor.from_pretrained(ROOT_PATH / FEATURE_EXTRACTOR)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            ROOT_PATH / SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        get_depth_map = DepthMapTransform(
                image_processor_cache=ROOT_PATH / IMAGE_PROCESSOR_CACHE, 
                depth_estimation_cache=ROOT_PATH / DEPTH_ESTIMATION_CACHE)
        get_canny_map = CannyMapTransform()
        face_detector = MediaPipeFaceMaskDetector()
        controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            cache_dir=ROOT_PATH / CONTROLNET_DEPTH_PATH,
            torch_dtype=torch.float16,
        )
        controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            cache_dir=ROOT_PATH / CONTROLNET_CANNY_PATH,
            torch_dtype=torch.float16,
        )
        self.landscapePipeline = SdxlBackgroundPipeline(
            safety_checker=safety_checker, 
            get_depth_map=get_depth_map,
            get_canny_map=get_canny_map,
            face_detector=face_detector,
            controlnet_depth=controlnet_depth,
            controlnet_canny=controlnet_canny,
            feature_extractor=feature_extractor,
            lora_model=ROOT_PATH / LORA_CACHE_CUSTOM,
            sdxl_model=ROOT_PATH / BASE_CACHE,
            refiner_model=ROOT_PATH / REFINER_MODEL_CACHE,
        ).setup()

        self.yolo = YoloPersonTransformator(yolo_cache=ROOT_PATH / YOLO_MODEL_CACHE)
        self.outfitPipeline = SdxlOutfitPipeline(
            safety_checker=safety_checker,
            yolo=self.yolo,
            feature_extractor=feature_extractor,
            lora_model=ROOT_PATH / LORA_CACHE_CUSTOM,
            sdxl_model=ROOT_PATH / BASE_CACHE,
            refiner_model=ROOT_PATH / REFINER_MODEL_CACHE,
        ).setup()

    @torch.inference_mode()
    def predict(
        self,
        landscape_prompt: str = Input(description="Landscape prompt", default=default_landscape_prompt),
        outfit_prompt: str = Input(description="Outfit prompt", default=default_outfit_prompt),
        camera_prompt: str = Input(description="Camera prompt", default=default_camera_prompt),
        negative_prompt: str = Input(
            description="Negative Input prompt", default="blur, cartoon, painting, drawing, illustration, 3D render, CGI, low resolution, abstract, unrealistic lighting, overexposed, underexposed, pixelated, oversaturated colors, grainy, smooth, flat, distorted, artistic, fantasy, bokeh, oil painting, watercolor, sketch, airbrush, anime, digital art, surreal, glowing edges, exaggerated details, unnatural textures, filter effects"
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        disable_face_recognition: bool = Input(
            description="Disable face recognition",
            default=False,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="DPM++2MSDE",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=50,
            default=50,
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "base_image_refiner"],
            default="base_image_refiner",
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=5,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            ge=0,
            le=50,
            default=15.8,
        ),
        lora_scale: float = Input(
            description="Scale for LORA",
            ge=-2,
            le=1,
            default=0,
        ),
        canny_condition_scale: float = Input(
            description="The bigger this number is, the more Canny interferes",
            default=0.3,
            ge=0.0,
            le=1.0,
        ),
        depth_condition_scale: float = Input(
            description="The bigger this number is, the more Depth interferes",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        strength: float = Input(
            description="The bigger this number is, the more image destruction",
            default=0.4,
            ge=0.0,
            le=1.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images",
            default=True,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        landscape_kwargs = {
            "canny_condition_scale" : canny_condition_scale, 
            "depth_condition_scale" : depth_condition_scale,
            "image" : image,
            "strength" : strength,
            "num_outputs" : num_outputs,
            "refine" : refine,
            "scheduler" : scheduler,
            "refine_steps" : refine_steps,
            "lora_scale": lora_scale,
            "landscape_prompt": landscape_prompt,
            "outfit_prompt": outfit_prompt,
            "camera_prompt": camera_prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "disable_face_recognition": disable_face_recognition,
            "disable_safety_checker": disable_safety_checker,
        }
        print("Running pipeline")
        output_paths = []
        landscape_result = self.landscapePipeline.predict(landscape_kwargs)
        outfit_kwargs = {
            "image" : landscape_result.output_image,
            "strength" : strength,
            "num_outputs" : num_outputs,
            "refine" : refine,
            "scheduler" : scheduler,
            "refine_steps" : refine_steps,
            "outfit_prompt": outfit_prompt,
            "camera_prompt": camera_prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "disable_safety_checker": disable_safety_checker,
            "faces_mask": landscape_result.faces_mask,
        }
        outfit_result = self.outfitPipeline.predict(outfit_kwargs)
        outputs = [landscape_result.output_image, outfit_result.output_image, outfit_result.mask]
        output_paths = []
        generation_id = uuid.uuid4()

        for output in outputs:
            file_id = uuid.uuid4()
            output_path = f"/tmp/output-{generation_id}-{file_id}.png"
            output.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths
        