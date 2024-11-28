import os
from pathlib import Path
import time

import numpy as np
import torch
from core.pipelines.pipeline import Pipeline
from core.transform.yolo_person_transformator import YoloPersonTransformator
from core.utils.image import combine_images, remove_background, resize_image
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    DiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor
from core.pipelines.schedulers import SCHEDULERS
from PIL import Image
from diffusers.image_processor import IPAdapterMaskProcessor



class OutfitResult:
    def __init__(self, output_image: Image, clothing_mask: Image):
        self.output_image = output_image
        self.clothing_mask = clothing_mask



class SdxlOutfitPipeline(Pipeline):
    #constructor
    def __init__(self, 
                 sdxl_model: Path,
                 lora_model: Path,
                 refiner_model: Path,
                 ip_adapter_model: Path,
                 yolo: YoloPersonTransformator,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 ):
        self.yolo = yolo
        self.safety_checker = safety_checker
        self.refiner_model = refiner_model
        self.sdxl_model = sdxl_model
        self.lora_model = lora_model
        self.ip_adapter_model = ip_adapter_model
        self.feature_extractor = feature_extractor
    def setup(self):
        print("Setup OutfitPipeline pipeline...")
        
        start = time.time()
        print(f"Loading StableDiffusionXLControlNetInpaintPipeline pipeline from path {self.sdxl_model}" )
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.sdxl_model,
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
        ).to("cuda")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", cache_dir=self.ip_adapter_model)
        print(f"Loading Refiner pipeline from path {self.refiner_model}" )
        self.refiner = DiffusionPipeline.from_pretrained(
            self.refiner_model,
            torch_dtype=torch.float16,
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            use_safetensors=True,
        ).to("cuda")

        print(f"setup OutfitPipeline took: {time.time() - start}")
        return self

    def predict(self, args: dict) -> OutfitResult:
        print(f"Using seed: {args['seed']}")
        seed = args['seed']
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        ip_adapter_scale = args['ip_adapter_scale']
        
        prompt = args['prompt'] + " " + args['camera_prompt']

        generator = torch.Generator("cuda").manual_seed(seed)
        
        # OOMs can leave vae in bad state
        if self.pipe.vae.dtype == torch.float32:
            self.pipe.vae.to(dtype=torch.float16)

        original_image = args['image']
        resized_image, width, height = resize_image(original_image)
        person_mask = self.yolo.isolate(resized_image)
        faces_mask = args['faces_mask']

        # masque sur les vetements et le visage
        face_mask_np = np.array(faces_mask)  
        person_mask_np = np.array(person_mask)
        clothing_mask_np = np.logical_and(person_mask_np, face_mask_np)
        clothing_mask_np = clothing_mask_np.astype(np.uint8) * 255
        clothing_mask = Image.fromarray(clothing_mask_np)
        

        ip_adapter_image = load_image(str(args['ip_adapter_image']))
        #ip_images = [[outfit_ip_adapter_image]]   
        #ip_adapter_mask = processor.preprocess([clothing_mask], height=height, width=width)
        #clothing_mask = [ip_adapter_mask.reshape(1, ip_adapter_mask.shape[0], ip_adapter_mask.shape[2], ip_adapter_mask.shape[3])]
        
        sdxl_kwargs = {}

        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        sdxl_kwargs["mask_image"] = clothing_mask
        sdxl_kwargs["image"] = resized_image
        sdxl_kwargs["ip_adapter_image"] = ip_adapter_image
        sdxl_kwargs["strength"] = args['strength']
        if args['refine'] == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"
        pipe = self.pipe
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        

        pipe.scheduler = SCHEDULERS[args['scheduler']].from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        print("Initialized scheduler")


        common_args = {
            "prompt": prompt * args['num_outputs'],
            "negative_prompt": [args['negative_prompt']] * args['num_outputs'],
            "guidance_scale": args['guidance_scale'],
            "generator": generator,
            "num_inference_steps": args['num_inference_steps'],
        }


        #self.pipe.fuse_lora(lora_scale=args['lora_scale'])
        #cross_attention_kwargs={"scale": args['lora_scale']}

        print("Running pipeline")
        output = pipe(**common_args, **sdxl_kwargs)

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
        return OutfitResult(output_image, clothing_mask=clothing_mask)
