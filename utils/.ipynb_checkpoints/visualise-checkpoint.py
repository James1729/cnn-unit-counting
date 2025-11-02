import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

from pathlib import Path
from glob import glob
from typing import List, Dict
from utils.tidy import get_unit_references
from matplotlib.transforms import Bbox
from matplotlib.transforms import TransformedBbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase

class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        sx, sy = self.image_stretch 
        bb = Bbox.from_bounds(
            xdescent - sx,
            ydescent - sy,
            width + sx,
            height + sy
        )
        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)

        return [image]

    def set_image(
        self, 
        image_path, 
        reshape_dimensions: List[int],
        num_channels: int,
        image_stretch=(0, 0), 
    ):
        self.image_data = read_and_decode(
            filename=image_path, 
            reshape_dims=reshape_dimensions, 
            num_channels=num_channels
        )        
        self.image_stretch = image_stretch


def show_sample_predictions(
    data: pd.DataFrame, 
    reshape_dimensions: List[int],
    num_channels: int,
    gen_key: str,
    info: Dict,
    num_obs: int = 10,
    assess_misfits: bool = False,
) -> None:
    if assess_misfits:
        plot_data = data.query("error != 0")
    else:
        plot_data = data.copy()
    
    target_unit_image_path = get_unit_references(
        faction=info["target_units"]["faction"], 
        colour=info["target_units"]["colour"], 
        unit=info["target_units"]["unit"],
    )[0]["filename"]

    if plot_data.shape[0] > 0:
        plot_data_subsample = plot_data.sample(num_obs)
        for _, row in plot_data_subsample.iterrows():
            print(row["sample_id"])
            sample_instance = row["sample_id"]
            prediction = row["prediction"]
            num_targets = row["num_targets"]

            custom_handler = ImageHandler()
            custom_handler.set_image(
                target_unit_image_path,
                # "raw_stock_images/processed/units/s_y_ht_3.png",
                # image_stretch=(0, 20)
                reshape_dimensions=reshape_dimensions,
                num_channels=num_channels,
                image_stretch=(0, 10)
            )
            im = show_image(
                filename=f"output/{gen_key}/images/{sample_instance}.png",
                # annotate_text=f"tank prediction={round(prediction)}",
                reshape_dimensions=reshape_dimensions,
                num_channels=num_channels,
            )
            plt.legend(
                [im],
                [f"is the target. Target prediction={round(prediction)} (actual targets={num_targets})"],
                handler_map={im: custom_handler},
                labelspacing=2,
                frameon=True,
                loc='upper center', bbox_to_anchor=(0.5, -0.025)
            )
            plt.show()
    else:
        print("no misfits in data given")


def show_image(
    filename: str,
    reshape_dimensions: List[int],
    num_channels: int,
):
    image = read_and_decode(
        filename=filename, 
        reshape_dims=reshape_dimensions,
        num_channels=num_channels,
    )
    ax = plt.imshow(image.numpy())
    plt.axis('off')
    return ax

def read_and_decode(filename: str, reshape_dims, num_channels: int):
    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image, channels=num_channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, reshape_dims)
