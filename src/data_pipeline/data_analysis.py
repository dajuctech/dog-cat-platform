import os
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_sample_images(generator, class_names, num_samples=5):
    """
    Plot sample images from the generator for EDA.
    """
    try:
        logger.info("Plotting sample images for EDA...")
        images, labels = next(generator)
        
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            ax = plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i])
            plt.title(f"Label: {class_names[int(labels[i])]}")
            plt.axis("off")
        plt.show()
        logger.info("Sample images plotted.")
    except Exception as e:
        logger.error(f"Error in EDA plotting: {e}")
        raise

if __name__ == "__main__":
    from data_preprocessing import create_generators
    
    TRAIN_DIR = 'data/processed/dogs-vs-cats-vvsmall/train'
    VALIDATION_DIR = 'data/processed/dogs-vs-cats-vvsmall/validation'
    train_gen, _ = create_generators(TRAIN_DIR, VALIDATION_DIR)
    class_names = list(train_gen.class_indices.keys())
    
    plot_sample_images(train_gen, class_names)
