import numpy as np
import pandas as pd
import uuid

from pathlib import Path
from joblib import Parallel, delayed
from PIL import Image
from random import choices, randint
from typing import List, Dict
from tqdm import tqdm

def determine_min_dim(background_image_params: List[Dict]) -> tuple[int, int]:
    min_dim = np.inf
    min_w, min_h = np.inf, np.inf
    for background_image_param in background_image_params:
        background_im = Image.open(background_image_param["path"])
        width, height = background_im.size
        if width*height < min_dim:
            min_dim = width*height
            min_w, min_h = width, height
    return min_w, min_h

def paste_images(
    background_image: Image,
    sampling_units: List[Dict],
    num_units_to_sample: int,
) -> Image:
    edge_buffer = 100
    units = [Image.open(unit["filename"]).resize(unit["resize"]) for unit in sampling_units]
    for sample in range(0, num_units_to_sample):
        sample_im = choices(units, k=1)[0]
        (x,y) = (randint(0, background_image.width-edge_buffer), randint(0, background_image.height-edge_buffer))
        background_image.paste(im=sample_im, box=(x,y), mask=sample_im) 
    return background_image

def sample_image(
    background_image_params: str,
    target_units: List[Dict],
    output_dir: str,
    gen_ref: str,
    noise_units: List[Dict],
    resize: List[int],
) -> None:
    sample_id = str(uuid.uuid4())
    target_path = f"{output_dir}/{gen_ref}/target"
    image_path = f"{output_dir}/{gen_ref}/images"

    background_param = choices(background_image_params, k=1)[0]
    background_im = Image.open(background_param["path"])
    target_unit_max = background_param["max_num_targets_per_sample"]
    noise_unit_max = background_param["max_num_noise_per_sample"]

    if len(noise_units) > 0:
        background_im = paste_images(
            background_image=background_im,
            sampling_units=noise_units,
            num_units_to_sample=randint(0, noise_unit_max),
        )
    num_targets = randint(0, target_unit_max)
    background_im = paste_images(
        background_image=background_im,
        sampling_units=target_units,
        num_units_to_sample=num_targets,
    )    
    pd.DataFrame({
        "sample_id": [sample_id], 
        "num_targets": [num_targets],         
    }).to_csv(f"{target_path}/{sample_id}.csv", index=False)
    background_im.resize(resize).save(f"{image_path}/{sample_id}.png")
        
def generate_sample_images(
    background_image_params: List[str],
    target_units: List[Dict],
    output_dir: str,
    gen_ref: str,
    num_image_samples: int,
    noise_units: List[Dict],
    resize: List[int],
) -> None:

    Path(f"{output_dir}/{gen_ref}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/{gen_ref}/target").mkdir(parents=True, exist_ok=True)
    
    Parallel(n_jobs=-1)(
        delayed(sample_image)(    
            background_image_params=background_image_params,
            target_units=target_units,
            output_dir=output_dir,
            gen_ref=gen_ref,
            noise_units=noise_units,
            resize=resize,
        ) for sample in tqdm(range(0, num_image_samples))
    )
