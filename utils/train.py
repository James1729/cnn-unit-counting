
import pandas as pd
import tensorflow as tf
import keras

from functools import partial
from pathlib import Path
from glob import glob
from typing import List
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
from tqdm import tqdm

class Models():
    def __init__(
        self,
        model_ref: str, 
        gen_ref: str,
        img_height: int,
        img_width: int,
        img_channels: int,
        batch_size: int,
        num_epochs: int,
        steps_per_epoch: int,
        train_proportion: float,
        validation_steps: int,
        learning_rate: float,
        patience: int,
    ):
        self.model_ref = model_ref
        self.gen_ref = gen_ref
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_proportion = train_proportion
        self.validation_steps = validation_steps
        self.model = self.get_model()
        self.learning_rate = learning_rate
        self.patience = patience

    def get_model(self) -> keras.Sequential: 
        if self.model_ref == "cnn1":
            return keras.Sequential([
                keras.Input(shape=(self.img_height, self.img_width, self.img_channels)),

                keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu', data_format="channels_last"),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                
                keras.layers.Flatten(),
                keras.layers.Dense(units=1, activation=None)
            ])
        if self.model_ref == "cnn14":
            return keras.Sequential([
                keras.Input(shape=(self.img_height, self.img_width, self.img_channels)),

                keras.layers.Conv2D(filters=300, kernel_size=(2, 2), activation='relu', data_format="channels_last"),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=350, kernel_size=(2, 2), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=450, kernel_size=(2, 2), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=550, kernel_size=(2, 2), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=900, kernel_size=(2, 2), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Conv2D(filters=1600, kernel_size=(2, 2), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),

                keras.layers.Flatten(),

                keras.layers.Dense(units=1, activation=None)
            ])
    def train_and_evaluate(self) -> None:
        print("Aggregating image ids & determining train/validation splits")
        train, evaluation = train_eval_csv(
            training_proportion=self.train_proportion, 
            gen_ref=self.gen_ref,
            model_ref=self.model_ref,
        )
        print(f"training, evaluation sizes: {train.shape[0]}, {evaluation.shape[0]}")
        train = train.sort_values(by="sample_id").reset_index().drop(columns="index")
        evaluation = evaluation.sort_values(by="sample_id").reset_index().drop(columns="index")

        self.model.summary()

        train_generator = partial(
            tensor_generator, 
            train, 
            [self.img_height, self.img_width], 
            self.img_channels,
            self.gen_ref,
            )
        eval_generator = partial(
            tensor_generator, 
            evaluation, 
            [self.img_height, self.img_width], 
            self.img_channels,
            self.gen_ref,
        )

        train_dataset = generate_dataset(
            gen=train_generator, 
            output_shape=(self.img_height, self.img_width, self.img_channels),
            batch_size=self.batch_size
        )
        eval_dataset = generate_dataset(
            gen=eval_generator,
            output_shape=(self.img_height, self.img_width, self.img_channels),
            batch_size=self.batch_size,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=self.patience, 
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"output/{self.gen_ref}/{self.model_ref}.best.weights.h5", 
                save_freq='epoch', 
                verbose=1, 
                monitor='val_loss', 
                save_weights_only=True, 
                save_best_only=True
            ) 
        ]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MeanAbsoluteError(),
        )
        self.model.fit( 
            train_dataset, 
            validation_data=eval_dataset,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=callbacks,
            validation_steps=self.validation_steps,
        )
        self.model.load_weights(f"output/{self.gen_ref}/{self.model_ref}.best.weights.h5")
        train, evaluation, results = results_summary(
            train_df=train, 
            eval_df=evaluation, 
            tf_model=self.model,
            reshape_dimensions=[self.img_height, self.img_width],   
            num_channels=self.img_channels,
            gen_ref=self.gen_ref,
            model_ref=self.model_ref,
        )
        print(f"fit results: {results}")

def generate_dataset(gen, output_shape: tuple[int], batch_size: int):
    return tf.data.Dataset.from_generator(
        generator=gen,
        output_signature=(
            tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    ).repeat().batch(batch_size)

def tensor_generator(
    data: pd.DataFrame,
    reshape_dimensions: List[int],
    num_channels: int,
    gen_ref: str,
    ): 
    n = data.shape[0]
    index = 0
    while index <= n-1:
        sample_instance = data.loc[index].sample_id
        num_targets = data.loc[index].num_targets
            
        filename = f"output/{gen_ref}/images/{sample_instance}.png"
        image = read_and_decode(
            filename=filename, 
            reshape_dims=reshape_dimensions,
            num_channels=num_channels,
        )
        yield image, num_targets
        index += 1

def get_sample_tensors(
    data: pd.DataFrame, 
    sample_instance: str,
    reshape_dimensions: List[int],
    num_channels: int,
    gen_ref: str,
): 
    slices = []
    responses = [] 
    
    num_targets = data.query(f"sample_id=='{sample_instance}'")["num_targets"]

    filename = f"output/{gen_ref}/images/{sample_instance}.png"

    image = read_and_decode(
        filename=filename, 
        reshape_dims=reshape_dimensions,
        num_channels=num_channels,
    )
    slices.append(image)
    responses.append(num_targets)
    return tf.data.Dataset.from_tensor_slices(([slices], [responses])) 

def read_csvs(read_csv_paths: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for csv_path in tqdm(read_csv_paths):
        df = pd.concat([df, pd.read_csv(csv_path)])
    return df

def train_eval_csv(
    training_proportion: float, 
    gen_ref: str,
    model_ref: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_paths = glob(f"output/{gen_ref}/target/*.csv")
    csv_paths.sort()
    output_path = f"output/{gen_ref}/target/aggregated/"

    num_training = int(len(csv_paths)*training_proportion)
    num_evaluation = len(csv_paths) - num_training
    train_csvs = csv_paths[0:num_training]
    evaluation_csvs = csv_paths[num_training:len(csv_paths)]
    
    train = read_csvs(read_csv_paths=train_csvs).sample(frac = 1).reset_index().drop(columns="index")
    evaluation = read_csvs(read_csv_paths=evaluation_csvs).sample(frac = 1).reset_index().drop(columns="index")
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(f"{output_path}/{model_ref}_train.csv", index=False)
    evaluation.to_csv(f"{output_path}/{model_ref}_evaluation.csv", index=False)

    return train, evaluation

def add_predictions(
    tf_model,
    df: pd.DataFrame, 
    reshape_dimensions: List[int],
    num_channels:int,
    gen_ref: str,
) -> pd.DataFrame:
    tqdm.pandas(desc="adding predictions to data")
    df["prediction"] = df[["sample_id"]].progress_apply(
        lambda x: round(tf_model.predict(
                    x=get_sample_tensors(
                        data=pd.read_csv(f"output/{gen_ref}/target/{x.sample_id}.csv"),
                        sample_instance=x.sample_id,
                        reshape_dimensions=reshape_dimensions,
                        num_channels=num_channels,
                        gen_ref=gen_ref,
                    ).get_single_element()[0],
                    verbose=0,
            )[0][0])
        , axis = 1
    )
    df["error"] = df["num_targets"] - df["prediction"]
    return df

def error_percentage(data: pd.DataFrame) -> float:
    return 100*data[data.error != 0].shape[0]/data.shape[0]

def accuracy_percentage(data: pd.DataFrame) -> float:
    return 100-error_percentage(data=data)

def results_summary(
    tf_model,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    reshape_dimensions: List[int],
    num_channels: int,
    gen_ref: str,
    model_ref: str,
):
    Path("results").mkdir(parents=True, exist_ok=True)
    print("prediction adding to train dataset...")
    train_df = add_predictions(
        df=train_df, 
        tf_model=tf_model,
        reshape_dimensions=reshape_dimensions,
        num_channels=num_channels,
        gen_ref=gen_ref,
    )
    print("concluded predictions added to train dataset")
    print("prediction adding to validation dataset...")
    eval_df = add_predictions(
        df=eval_df, 
        tf_model=tf_model,
        reshape_dimensions=reshape_dimensions,
        num_channels=num_channels,
        gen_ref=gen_ref,
    )
    print("concluded predictions added to validation dataset")
    train_mae = mean_absolute_error(y_true=train_df["num_targets"], y_pred=train_df["prediction"])
    eval_mae = mean_absolute_error(y_true=eval_df["num_targets"], y_pred=eval_df["prediction"])

    accuracy_pct_train,accuracy_pct_eval = accuracy_percentage(data=train_df), accuracy_percentage(data=eval_df)
    results = pd.DataFrame({
        "set": ["train", "evaluation"],
        "mae": [train_mae, eval_mae],
        "accuracy %": [accuracy_pct_train, accuracy_pct_eval],
    })
    train_df.to_csv(f"results/{gen_ref}-{model_ref}_train.csv", index=False)
    eval_df.to_csv(f"results/{gen_ref}-{model_ref}_evaluation.csv", index=False)
    results.to_csv(f"results/{gen_ref}-{model_ref}_results.csv", index=False)

    return train_df, eval_df, results

def read_and_decode(filename: str, reshape_dims, num_channels: int):
    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image, channels=num_channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, reshape_dims)