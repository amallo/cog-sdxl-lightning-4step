import os
from pathlib import Path
import time

import numpy as np

from core.transform.media_pipe_face_mask_detector import MediaPipeFaceMaskDetector
import torch
from core.transform.canny_map_transform import CannyMapTransform
from core.pipelines.pipeline import Pipeline
from core.transform.depth_map_transform import DepthMapTransform
from core.transform.yolo_person_transformator import YoloPersonTransformator
from core.utils.image import load_image_from_path, resize_image
from paths import BASE_CACHE, FEATURE_EXTRACTOR, LORA_CACHE_CUSTOM, REFINER_MODEL_CACHE

from diffusers import (
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    DiffusionPipeline,
    ControlNetModel,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from transformers import CLIPImageProcessor
from core.pipelines.schedulers import SCHEDULERS
from PIL import Image
from diffusers.image_processor import IPAdapterMaskProcessor

class LandscapeResult:
    def __init__(self, output_image: Image, faces_mask: Image, canny: Image, depth: Image):
        self.output_image = output_image
        self.faces_mask = faces_mask
        self.canny = canny
        self.depth = depth



class SdxlBackgroundAndClothsPipeline(Pipeline):
    #constructor
    def __init__(self, 
                 lora_model: Path,
                 sdxl_model: Path,
                 refiner_model: Path,
                 ip_adapter_model: Path,
                 safety_checker: StableDiffusionSafetyChecker, 
                 get_depth_map: DepthMapTransform, 
                 get_canny_map: CannyMapTransform,
                 face_detector: MediaPipeFaceMaskDetector,
                 controlnet_depth: ControlNetModel,
                 controlnet_canny: ControlNetModel,
                 yolo: YoloPersonTransformator,
                 feature_extractor: CLIPImageProcessor):
        self.yolo = yolo
        self.safety_checker = safety_checker
        self.get_depth_map = get_depth_map
        self.get_canny_map = get_canny_map
        self.face_detector = face_detector
        self.controlnet_depth = controlnet_depth
        self.controlnet_canny = controlnet_canny
        self.feature_extractor = feature_extractor
        self.refiner_model = refiner_model
        self.sdxl_model = sdxl_model
        self.lora_model = lora_model
        self.ip_adapter_model = ip_adapter_model
    def setup(self):
        
        start = time.time()
        print(f"Loading StableDiffusionXLControlNetInpaintPipeline pipeline from path {self.sdxl_model}"  )
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            self.sdxl_model,
            torch_dtype=torch.float16,
            controlnet=[self.controlnet_canny, self.controlnet_depth],
            local_files_only=True,
            use_safetensors=True,
        ).to("cuda")
        self.pipeClothing = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.sdxl_model,
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
        ).to("cuda")
        self.pipeClothing.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", cache_dir=self.ip_adapter_model)
        print(f"Loading Refiner pipeline from path {self.refiner_model}" )
        self.refiner = DiffusionPipeline.from_pretrained(
            self.refiner_model,
            torch_dtype=torch.float16,
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            use_safetensors=True,
        ).to("cuda")


        print(f"Loading LORA model from path {self.lora_model}")    
        self.pipe.load_lora_weights(self.lora_model, weight_name="lora.safetensors")
        print(f"setup  SdxlBackgroundPipeline took: {time.time() - start}")
        return self

    def predict(self, args: dict) -> LandscapeResult:
        print(f"Using seed: {args['seed']}")
        seed = args['seed']
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        enable_freeu = args['enable_freeu']

        
        prompt = args['prompt'] + " " + args['outfit_prompt'] + " " + args['camera_prompt']

        generator = torch.Generator("cuda").manual_seed(seed)

        ip_adapter_scale = args['outfit_ip_adapter_scale']
        ip_adapter_image = args['outfit_ip_adapter_image']
        
     
        # OOMs can leave vae in bad state
        if self.pipe.vae.dtype == torch.float32:
            self.pipe.vae.to(dtype=torch.float16)


        loaded_image = load_image_from_path(args['image'])
        resized_image, width, height = resize_image(loaded_image)
    

        depth_image =self.get_depth_map(resized_image)
        canny_image = self.get_canny_map(resized_image)
        faces_mask = self.face_detector(resized_image)
        images = [canny_image, depth_image]
        print("Created canny and depth maps")   

        person_mask = self.yolo.isolate(resized_image)
        face_mask_np = np.array(faces_mask)  
        person_mask_np = np.array(person_mask)
        clothing_mask_np = np.logical_and(person_mask_np, face_mask_np)
        clothing_mask_np = clothing_mask_np.astype(np.uint8) * 255
        clothing_mask = Image.fromarray(clothing_mask_np)


        sdxl_kwargs = {}



        
        print(f"Prompt: {prompt}")
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        sdxl_kwargs["controlnet_conditioning_scale"] = [args['canny_condition_scale'], args['depth_condition_scale']]
        sdxl_kwargs["mask_image"] = faces_mask
        sdxl_kwargs["image"] = resized_image
        sdxl_kwargs["control_image"] = images
        sdxl_kwargs["strength"] = args['strength']
        sdxl_kwargs["ip_adapter_image"] = ip_adapter_image
        if args['refine'] == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"
        pipe = self.pipe
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        if (enable_freeu):
            self.pipe.enable_freeu(
                s1=0.9,  # Facteur d'échelle pour les features de bas niveau
                s2=0.2,  # Facteur d'échelle pour les features de haut niveau
                b1=1.3,  # Facteur de boost pour les features de bas niveau
                b2=1.4   # Facteur de boost pour les features de haut niveau
            )
        else:
            print("FreeU is disabled")  
            self.pipe.disable_freeu()

        pipe.scheduler = SCHEDULERS[args['scheduler']].from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        print("Initialized scheduler")


        common_args = {
            "prompt": [prompt] * args['num_outputs'],
            "negative_prompt": [args['negative_prompt']] * args['num_outputs'],
            "guidance_scale": args['guidance_scale'],
            "generator": generator,
            "num_inference_steps": args['num_inference_steps'],
        }


        self.pipe.fuse_lora(lora_scale=args['lora_scale'])
        cross_attention_kwargs={"scale": args['lora_scale'], "ip_adapter_masks": clothing_mask}

        print("Running pipeline")
        #self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4) # Enable FreeU for quality
        output = pipe(**common_args, **sdxl_kwargs, **cross_attention_kwargs)

        if args['refine'] == "base_image_refiner":
            refiner_kwargs = {
                "image": output.images,
            }

            common_args_without_dimensions = {
                k: v for k, v in common_args.items() if k not in ["width", "height"]
            }

            if args['refine'] == "base_image_refiner" and args['refine_steps']:
                common_args_without_dimensions["num_inference_steps"] = args['refine_steps']

            output = self.refiner(**common_args_without_dimensions, **refiner_kwargs)

        if not args['disable_safety_checker']:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_image: Image = None
        for i, image in enumerate(output.images):
            if not args['disable_safety_checker']:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_image = image

        if output_image is None:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )
        return LandscapeResult(output_image, faces_mask=faces_mask, canny=canny_image, depth=depth_image)