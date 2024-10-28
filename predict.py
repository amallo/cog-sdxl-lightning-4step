# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import shutil
import subprocess
import numpy as np
from typing import List
from transformers import CLIPImageProcessor, DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    DiffusionPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    ControlNetModel,
)
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
import cv2
from rembg import new_session, remove

UNET = "sdxl_lightning_4step_unet.pth"
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_DEPTH_PATH = "controlnet-depth"
CONTROLNET_CANNY_PATH = "controlnet-canny"
UNET_CACHE = "unet-cache"
BASE_CACHE = "checkpoints"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "feature-extractor"
MODEL_NORMAL = "prs-eth/marigold-normals-lcm-v0-1"
IMAGE_PROCESSOR_CACHE = "image-processor-cache"
DEPTH_ESTIMATION_CACHE = "depth-estimation-cache"
REFINER_URL = "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
REFINER_MODEL_CACHE = "./refiner-cache"
#MODEL_URL = "https://weights.replicate.delivery/default/sdxl-lightning/sdxl-1.0-base-lightning.tar"
MODEL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
UNET_URL = "https://weights.replicate.delivery/default/comfy-ui/unet/sdxl_lightning_4step_unet.pth.tar"
LORA_WEIGHTS = "https://replicate.delivery/pbxt/SqRHCOhHkOoGJ5mW93YW0F863hV6IJ1vf96RvI5v65cWofkTA/trained_model.tar"
LORA_CACHE_CUSTOM = "./trained-model/custom"
LORA_CACHE_POSTAPO = "./trained-model/postapo"

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DPM++2MSDE": KDPM2AncestralDiscreteScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

default_prompt="""
The city is now completely abandoned, a ghost town overtaken by nature. 
Buildings are in ruins, collapsing under their own weight, covered with tags, bullet holes. 
The streets are cracked, with plants and vines pushing through the asphalt and creeping up the walls. 
Trees and tall grass have grown wildly in the middle of the once-bustling roads. No humans are in sight, 
and the city is eerily quiet except for the sound of wind rustling through the leaves. 
The few remaining cars are rusting, partially swallowed by the greenery.
""" 
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        print("Loading model")
        if not os.path.exists(BASE_CACHE):
            download_weights(MODEL_URL, BASE_CACHE)
        print("Loading SDXL refiner pipeline...")
        if not os.path.exists(REFINER_MODEL_CACHE):
            download_weights(REFINER_URL, REFINER_MODEL_CACHE)
        print("Loading Unet")
        #if not os.path.exists(UNET_CACHE):
        #    download_weights(UNET_URL, UNET_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        #print("Loading lora")
        #if not os.path.exists(LORA_CACHE):
        #    download_weights(LORA_WEIGHTS, LORA_CACHE)

        print("Loading depth transformer pipeline...")
        self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_CACHE,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")
        self.image_processor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=IMAGE_PROCESSOR_CACHE)
        self.depth_estimation = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=DEPTH_ESTIMATION_CACHE).to("cuda")
        self.refiner = DiffusionPipeline.from_pretrained(
            REFINER_MODEL_CACHE,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            vae=self.txt2img_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        self.controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            cache_dir=CONTROLNET_DEPTH_PATH,
            torch_dtype=torch.float16,
        )

        self.controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            cache_dir=CONTROLNET_CANNY_PATH,
            torch_dtype=torch.float16,
        )
 
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
        print("Loading txt2img pipeline...")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            BASE_CACHE,
            torch_dtype=torch.float16,
            controlnet=[self.controlnet_depth, self.controlnet_canny],
            local_files_only=True,
        ).to("cuda")
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        self.pipe.load_lora_weights(LORA_CACHE_CUSTOM, weight_name="lora.safetensors")
        
        #unet_path = os.path.join(UNET_CACHE, UNET)
        #self.pipe.unet.load_state_dict(torch.load(unet_path, map_location="cuda"))
        print("setup took: ", time.time() - start)

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept
    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    
    def get_depth_map(self, image):
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
    
    def get_canny_map(self, image):       
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image
    
    def remove_background(self, image):
        output = remove(image, only_mask=True)
        return Image.eval(output, lambda x: 255 - x)

    
    def resize_image(self, image):
        image_width, image_height = image.size
        print("Original width:"+str(image_width)+", height:"+str(image_height))
        new_width, new_height = self.resize_to_allowed_dimensions(image_width, image_height)
        print("new_width:"+str(new_width)+", new_height:"+str(new_height))
        image = image.resize((new_width, new_height))
        return image, new_width, new_height
    
    def get_width_height(self, width, height):
        width = (width//8)*8
        height = (height//8)*8
        return width,height 

    def resize_image_for_upscale(self, img_path,upscale_times):
        img             = load_image(img_path)
        if upscale_times <=0:
            return img
        width,height    = img.size
        width           = width * upscale_times
        height          = height * upscale_times
        width,height    = self.get_width_height(int(width),int(height))
        img             = img.resize(
            (width,height)
            ,resample = Image.LANCZOS if upscale_times > 1 else Image.AREA
        )
        return img
    
    def resize_to_allowed_dimensions(self, width, height):
        """
        Function re-used from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=default_prompt),
        negative_prompt: str = Input(
            description="Negative Input prompt", default="blur, cartoon, painting, drawing, illustration, 3D render, CGI, low resolution, abstract, unrealistic lighting, overexposed, underexposed, pixelated, oversaturated colors, grainy, smooth, flat, distorted, artistic, fantasy, bokeh, oil painting, watercolor, sketch, airbrush, anime, digital art, surreal, glowing edges, exaggerated details, unnatural textures, filter effects"
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        ip_adapter_image: Path = Input(
            description="Ip adapter image",
            default=None,
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
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=50,
            default=20,
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=4,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            ge=0,
            le=50,
            default=15.8,
        ),
        lora_scale: float = Input(
            description="Scale for LORA",
            ge=0,
            le=1,
            default=1.0,
        ),
        ip_adapter_scale: float = Input(
            description="Scale for ip adapter guidance",
            ge=0,
            le=1,
            default=0.2,
        ),
        canny_condition_scale: float = Input(
            description="The bigger this number is, the more Canny interferes",
            default=0.1,
            ge=0.0,
            le=1.0,
        ),
        depth_condition_scale: float = Input(
            description="The bigger this number is, the more Depth interferes",
            default=0.3,
            ge=0.0,
            le=1.0,
        ),
        strength: float = Input(
            description="The bigger this number is, the more image destruction",
            default=0.5,
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
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # OOMs can leave vae in bad state
        if self.pipe.vae.dtype == torch.float32:
            self.pipe.vae.to(dtype=torch.float16)



        loaded_image = self.load_image(image)
        resized_image, width, height = self.resize_image(loaded_image)
        

        depth_image =self.get_depth_map(resized_image)
        canny_image = self.get_canny_map(resized_image)
        mask_image = self.remove_background(resized_image)
        loaded_ip_adapter_image = self.load_image(ip_adapter_image)
        print("Created mask and maps")   


        sdxl_kwargs = {}
        print(f"Prompt: {prompt}")
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        sdxl_kwargs["controlnet_conditioning_scale"] = [depth_condition_scale, canny_condition_scale]
        sdxl_kwargs["image"] = resized_image
        sdxl_kwargs["mask_image"] = mask_image
        sdxl_kwargs["ip_adapter_image"] = loaded_ip_adapter_image
        sdxl_kwargs["control_image"] = [depth_image, canny_image]
        sdxl_kwargs["strength"] = strength
        if refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"
        pipe = self.pipe
        pipe.set_ip_adapter_scale(ip_adapter_scale)

        pipe.scheduler = SCHEDULERS[scheduler].from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        print("Initialized scheduler")


        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }


        self.pipe.fuse_lora(lora_scale=lora_scale)
        cross_attention_kwargs={"scale": lora_scale}

        print("Running pipeline")
        #self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4) # Enable FreeU for quality
        output = pipe(**common_args, **sdxl_kwargs, **cross_attention_kwargs)

        if refine == "base_image_refiner":
            refiner_kwargs = {
                "image": output.images,
            }

            common_args_without_dimensions = {
                k: v for k, v in common_args.items() if k not in ["width", "height"]
            }

            if refine == "base_image_refiner" and refine_steps:
                common_args_without_dimensions["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args_without_dimensions, **refiner_kwargs)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        mask_path=f"/tmp/mask.png"
        mask_image.save(mask_path)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        output_paths.append(Path(mask_path))
        return output_paths
