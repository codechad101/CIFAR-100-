#!/usr/bin/env python3
"""
CIFAR-100 CNN from Scratch - Python Script Version
Train a CNN with data augmentation, LR scheduler, early stopping.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pickle
import json

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    num_classes = 100
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test

def create_data_augmentation():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )

def create_cnn_model(input_shape=(32, 32, 3), num_classes=100):
    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_callbacks():
    return [
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('best_cnn_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

def train_model():
    x_train, x_test, y_train, y_test = load_and_preprocess_data()
    datagen = create_data_augmentation()
    datagen.fit(x_train)

    model = create_cnn_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')])

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=len(x_train)//32,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=create_callbacks(),
        verbose=1
    )

    if os.path.exists('best_cnn_model.h5'):
        model = tf.keras.models.load_model('best_cnn_model.h5')

    results = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]*100:.2f}%, Top-5 Accuracy: {results[2]*100:.2f}%")

    # Save results
    with open('cnn_results.json', 'w') as f:
        json.dump({
            'test_loss': float(results[0]),
            'test_accuracy': float(results[1]),
            'test_top5_accuracy': float(results[2])
        }, f, indent=2)

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    # Setup GPU memory growth if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    train_model()
