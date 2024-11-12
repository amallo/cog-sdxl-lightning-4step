import os
import time
import subprocess
from paths import DEPTH_ESTIMATION_CACHE, IMAGE_PROCESSOR_CACHE, YOLO_MODEL_CACHE, REFINER_MODEL_CACHE, REFINER_URL, SAFETY_CACHE, SAFETY_URL
from transformers import  DPTFeatureExtractor, DPTForDepthEstimation
from pathlib import Path

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
    