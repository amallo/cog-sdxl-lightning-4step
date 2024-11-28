# Obtenir le chemin racine du projet de manière dynamique
import os
from pathlib import Path
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import subprocess
import time
from core.cache.download_weights import DownloadWeights
import torch
sys.path.extend(['/IP-Adapter'])
from diffusers import ControlNetModel

ROOT_DIR_PATH = Path(ROOT_DIR)

# Make cache folder
"""
if not os.path.exists("model-cache"):
    os.makedirs("model-cache")


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

if not os.path.exists(BASE_CACHE):
    print("Loading SDXL model")
    download_weights(MODEL_URL, BASE_CACHE)
        
DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir=IMAGE_PROCESSOR_CACHE)
"""
download_weights = DownloadWeights(ROOT_DIR_PATH)
download_weights()