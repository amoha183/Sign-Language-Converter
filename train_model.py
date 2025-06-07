"""
Train Model Script for Sign Language Recognition

This script prepares the dataset, defines a VGG-style CNN model, and trains it for sign language recognition.
It includes data splitting, augmentation, model checkpointing, and training history saving.
"""
import os, shutil, numpy as np, tensorflow as tf
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D,
                                     GlobalAveragePooling2D, Dense, Dropout, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. DATA SPLIT (identical to yours)
# ------------------------------------------------------------------
def create_train_val_split(source_dir='data', train_dir='data/train',
                           val_dir='data/validation', val_split=0.2):
    # Split the dataset into training and validation sets
    os.makedirs(train_dir, exist_ok=True); os.makedirs(val_dir, exist_ok=True)
    for letter in os.listdir(source_dir):
        letter_path = os.path.join(source_dir, letter)
        if letter in ['train', 'validation'] or not os.path.isdir(letter_path):
            continue
        os.makedirs(os.path.join(train_dir, letter), exist_ok=True)
        os.makedirs(os.path.join(val_dir, letter), exist_ok=True)

        imgs = [f for f in os.listdir(letter_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:
            print(f"[warn] No images in {letter_path}"); continue
        tr, val = train_test_split(imgs, test_size=val_split, random_state=42, stratify=None)
        for im in tr:  shutil.copy2(os.path.join(letter_path, im), os.path.join(train_dir, letter, im))
        for im in val: shutil.copy2(os.path.join(letter_path, im), os.path.join(val_dir, letter, im))
        print(f"{letter:2}: {len(tr):4} train | {len(val):3} val")

# ------------------------------------------------------------------
# 2. MODEL – deeper VGG-style w/ GAP
# ------------------------------------------------------------------
def conv_block(x, filters, n_conv=2):
    # Convolutional block with BatchNorm and MaxPooling
    for _ in range(n_conv):
        x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    return MaxPooling2D(2,2)(x)

def create_model(input_shape=(400,400,3), n_classes=28):
    # Build a VGG-style CNN model for sign language recognition
    inputs = Input(shape=input_shape)

    x = conv_block(inputs, 64)          # 64-128-256-512 resembles VGG19
    x = conv_block(x, 128)
    x = conv_block(x, 256, n_conv=3)
    x = conv_block(x, 512, n_conv=3)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# ------------------------------------------------------------------
# 3. DATA GENERATORS – stronger aug
# ------------------------------------------------------------------
def create_generators(train_dir, val_dir, img_size=(400,400), bs=32):
    # Create data generators with augmentation for training and validation
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25, width_shift_range=0.25, height_shift_range=0.25,
        shear_range=0.15, zoom_range=0.25, horizontal_flip=True,
        brightness_range=[0.75,1.25], fill_mode='nearest'
    )
    # validation: only rescaling
    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(train_dir, target_size=img_size,
                                              class_mode='categorical', batch_size=bs, shuffle=True)
    val_gen   = val_aug.flow_from_directory(val_dir,   target_size=img_size,
                                            class_mode='categorical', batch_size=bs, shuffle=False)
    return train_gen, val_gen

# ------------------------------------------------------------------
# 4. TRAINING LOOP
# ------------------------------------------------------------------
def train_model():
    # Main training loop: prepares data, builds model, trains, and saves everything
    create_train_val_split()                          #  ❑ comment out if already split

    # OPTIONAL: enable mixed-precision for RTX/Apple-Silicon GPUs
    # from tensorflow.keras.mixed_precision import set_global_policy
    # set_global_policy('mixed_float16')

    strategy = tf.device('/GPU:0')  # single-GPU; adapt for multi-GPU with tf.distribute
    with strategy:
        model = create_model()

    # loss with label-smoothing
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    train_gen, val_gen = create_generators('data/train', 'data/validation', bs=32)

    ckpt_name = 'sign_converter_best.h5'
    callbacks = [
        ModelCheckpoint(ckpt_name, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_gen, epochs=8,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        callbacks=callbacks
    )

    model.save('checkpoint_sign_converter_alkadya.keras')
    np.save('training_history.npy', history.history)
    print("✓ Training complete.")
    return history

if __name__ == "__main__":
    hist = train_model()
