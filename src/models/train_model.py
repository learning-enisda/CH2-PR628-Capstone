import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

zip_file_path = 'data/raw/bsindo_training.zip'
extracted_dir = 'data/processed/train'

os.makedirs(extracted_dir, exist_ok=True)

if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
else:
    raise FileNotFoundError(f"The specified zip file '{zip_file_path}' does not exist.")

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

training_generator = datagen.flow_from_directory(
    directory=extracted_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    seed=42
)

num_classes = len(training_generator.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    loss=tf.losses.CategoricalCrossentropy(),
    optimizer=tf.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(
    training_generator,
    epochs=50,
    verbose=2
)
