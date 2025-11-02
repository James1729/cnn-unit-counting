import argparse
import tensorflow as tf

from utils.generate import generate_sample_images, determine_min_dim
from utils.train import Models
from utils.tidy import load_json, write_json, get_unit_references


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=str)
parser.add_argument("--train_reference", type=str, required=False)
parser.add_argument("--generate_reference", type=str, required=True)
args = parser.parse_args()
step = args.step
train_ref = args.train_reference
generate_ref = args.generate_reference

if step == "generate":
    generate_config = load_json(filename=f"configs/generate/{generate_ref}.json")
    width_min, height_min = determine_min_dim(background_image_params=generate_config["backgrounds"])
    generate_sample_images(
        background_image_params=generate_config["backgrounds"],
        target_units=get_unit_references(**generate_config["target_units"][0]),
        output_dir="output",
        gen_ref=generate_ref,
        num_image_samples=generate_config["num_samples"],
        noise_units=[] if not "noise_units" in generate_config.keys() else get_unit_references(**generate_config["noise_units"][0]),
        resize=[height_min, width_min],
    )
    write_json(
        data={
            "min_dim": [width_min, height_min], 
            "channels": 4,
            "target_units": generate_config["target_units"][0],
        }, 
        filename=f"output/{generate_ref}/generation_info.json",
    )

if step == "train":
    print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(f"tensorflow version: {tf.__version__}")
    generation_info = load_json(filename=f"output/{generate_ref}/generation_info.json")
    train_config = load_json(filename=f"configs/train/{train_ref}.json")

    IMG_WIDTH, IMG_HEIGHT = generation_info["min_dim"] 
    IMG_CHANNELS = generation_info["channels"]

    def override_dim(
        current_dims: tuple[int, int],
        proposal_dims: tuple[int, int],
    ) -> tuple[int, int]:
        if proposal_dims[0]*proposal_dims[1] < current_dims[0]*current_dims[1]:
            return proposal_dims
        else:
            return current_dims

    if "dim_override" in train_config.keys():
        IMG_WIDTH, IMG_HEIGHT = override_dim(
            current_dims=(IMG_WIDTH, IMG_HEIGHT),
            proposal_dims=(300, 300),
        )

    model = Models(
        model_ref=train_ref, 
        gen_ref=generate_ref,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        img_channels=IMG_CHANNELS,
        batch_size=train_config["batch_size"],
        num_epochs=train_config["num_epochs"],
        steps_per_epoch=train_config["steps_per_epoch"],
        train_proportion=train_config["train_proportion"],
        validation_steps=train_config["validation_steps"],
        learning_rate=train_config["learning_rate"],
        patience=train_config["patience"],

    )
    model.train_and_evaluate()

if __name__ == '__main__':
    print("running")
