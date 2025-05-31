import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_generators(train_dir, validation_dir, img_size=(300, 300), batch_size=32):
    """
    Create ImageDataGenerators with augmentation and rescaling.
    """
    try:
        logger.info("Initializing data generators...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data generator
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow training images
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        
        # Flow validation images
        validation_generator = val_datagen.flow_from_directory(
            validation_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        
        logger.info("Data generators created successfully.")
        return train_generator, validation_generator

    except Exception as e:
        logger.error(f"Error in creating data generators: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    TRAIN_DIR = 'data/processed/dogs-vs-cats-vvsmall/dogs-vs-cats-vvsmall/train'
    VALIDATION_DIR = 'data/processed/dogs-vs-cats-vvsmall/dogs-vs-cats-vvsmall/validation'
    train_gen, val_gen = create_generators(TRAIN_DIR, VALIDATION_DIR)
