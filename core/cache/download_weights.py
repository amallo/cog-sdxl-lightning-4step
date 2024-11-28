import os
import time
import subprocess

import torch
from paths import MODEL_URL, DEPTH_ESTIMATION_CACHE, BASE_CACHE, IMAGE_PROCESSOR_CACHE, IP_ADAPTER_CACHE, REFINER_MODEL_CACHE, REFINER_URL, SAFETY_CACHE, SAFETY_URL
from transformers import  DPTFeatureExtractor, DPTForDepthEstimation
from pathlib import Path
from diffusers import StableDiffusionXLPipeline


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class DownloadWeights:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
    def __call__(self):
         if not os.path.exists(self.root_dir/SAFETY_CACHE):
            print("Loading safety checker...")
            download_weights(SAFETY_URL, self.root_dir/SAFETY_CACHE)
         if not os.path.exists(self.root_dir/REFINER_MODEL_CACHE):
            print("Loading SDXL refiner pipeline...")
            download_weights(REFINER_URL, self.root_dir/REFINER_MODEL_CACHE)
         if not os.path.exists(self.root_dir / IMAGE_PROCESSOR_CACHE):
            print("Loading SDXL refiner pipeline...")
            DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=self.root_dir/IMAGE_PROCESSOR_CACHE)
         if not os.path.exists(self.root_dir/DEPTH_ESTIMATION_CACHE):
            print("Loading SDXL refiner pipeline...")
            DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=self.root_dir/DEPTH_ESTIMATION_CACHE)
         if not os.path.exists(self.root_dir/BASE_CACHE):
            print("Loading SDXL base pipeline...")
            download_weights(MODEL_URL, self.root_dir/BASE_CACHE)
         if not os.path.exists(self.root_dir/IP_ADAPTER_CACHE):
            print("Loading IP ADapter pipeline...")
            sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
               "stabilityai/stable-diffusion-xl-base-1.0",
               torch_dtype=torch.float16,
               variant="fp16",
               use_safetensors=True,
               cache_dir=self.root_dir/BASE_CACHE
            )
            sd_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", cache_dir=self.root_dir/IP_ADAPTER_CACHE)
