reform both code to captureed both

*import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
import logging
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2B0, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(input_shape=(300, 300, 3), num_classes=1, architecture='efficientnetb0'):
    """
    Build a transfer learning model with specified architecture.
    """
    logger.info(f"Building model using {architecture}...")
    
    if architecture == 'efficientnetb0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif architecture == 'efficientnetv2':
        base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif architecture == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported architecture. Choose 'efficientnetb0', 'efficientnetv2', or 'resnet50'.")
    
    base_model.trainable = False  # Freeze base model initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    logger.info("Model built successfully.")
    return model

def train_model(train_generator, val_generator, epochs=10, model_save_path='models/model.h5', architecture='efficientnetb0'):
    """
    Train the model with specified architecture.
    """
    logger.info(f"Starting training with {architecture}...")
    model = build_model(architecture=architecture)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
        ModelCheckpoint(model_save_path, save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    logger.info(f"Training completed. Model saved to {model_save_path}")
    return model, history

def evaluate_model(model, val_generator):
    """
    Evaluate the model and plot confusion matrix and ROC curve.
    """
    logger.info("Evaluating model...")
    val_generator.reset()
    preds = model.predict(val_generator)
    preds_binary = (preds > 0.5).astype(int).reshape(-1)
    true_labels = val_generator.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, preds_binary))

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds_binary)
    print("\nConfusion Matrix:")
    print(cm)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.colorbar()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    from src.data_pipeline.data_preprocessing import create_generators

    TRAIN_DIR = 'data/processed/dogs-vs-cats-vvsmall/train'
    VALIDATION_DIR = 'data/processed/dogs-vs-cats-vvsmall/validation'
    train_gen, val_gen = create_generators(TRAIN_DIR, VALIDATION_DIR)

    os.makedirs('models', exist_ok=True)
    model, history = train_model(train_gen, val_gen, epochs=10, architecture='efficientnetv2')

    # Evaluate after training
    evaluate_model(model, val_gen) *

*import os
import sys
import logging
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(input_shape=(300, 300, 3), num_classes=1):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

def train_with_mlflow(train_gen, val_gen, epochs=10, model_save_path='models/model_mlflow.h5'):
    logger.info("Starting training with MLflow tracking...")
    
    mlflow.set_experiment("dog-cat-classification")
    
    with mlflow.start_run():
        model = build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
            ModelCheckpoint(model_save_path, save_best_only=True)
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks
        )

        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", train_gen.batch_size)
        
        # Log metrics
        val_loss, val_acc = model.evaluate(val_gen)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)

        # Log model
        mlflow.keras.log_model(model, "model")

        logger.info("Training complete with MLflow tracking.")

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.data_pipeline.data_preprocessing import create_generators

    TRAIN_DIR = 'data/processed/dogs-vs-cats-vvsmall/train'
    VALIDATION_DIR = 'data/processed/dogs-vs-cats-vvsmall/validation'
    train_gen, val_gen = create_generators(TRAIN_DIR, VALIDATION_DIR)

    os.makedirs('models', exist_ok=True)
    train_with_mlflow(train_gen, val_gen, epochs=10) *