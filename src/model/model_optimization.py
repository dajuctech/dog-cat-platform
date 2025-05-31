import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from kerastuner import HyperModel, RandomSearch
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data_pipeline.data_preprocessing import create_generators

class CNNHyperModel(HyperModel):
    def build(self, hp):
        base_model_choice = hp.Choice('base_model', ['efficientnetb0', 'resnet50'])
        if base_model_choice == 'efficientnetb0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        else:
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float('dropout', 0.3, 0.7, step=0.1))(x)
        x = layers.Dense(hp.Int('dense_units', 64, 512, step=64),
                         activation='relu',
                         kernel_regularizer=regularizers.l2(hp.Choice('l2', [1e-4, 1e-3, 1e-2])))(x)
        output = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

def tune_hyperparameters(train_gen, val_gen):
    tuner = RandomSearch(
        CNNHyperModel(),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='dog_cat_tuning'
    )

    tuner.search(train_gen,
                 validation_data=val_gen,
                 epochs=5)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters()[0]

    print(f"Best hyperparameters: {best_hps.values}")
    return best_model

if __name__ == "__main__":
    TRAIN_DIR = 'data/processed/dogs-vs-cats-vvsmall/train'
    VAL_DIR = 'data/processed/dogs-vs-cats-vvsmall/validation'
    train_gen, val_gen = create_generators(TRAIN_DIR, VAL_DIR)

    best_model = tune_hyperparameters(train_gen, val_gen)
    best_model.save('models/best_model_tuned.h5')
