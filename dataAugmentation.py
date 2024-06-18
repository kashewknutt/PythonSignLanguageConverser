import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Define the path to your dataset
dataset_directory = 'indianSignLanguageDataset'

# Define image size and batch size
image_size = (64, 64)  # You can adjust the size based on your requirements
batch_size = 32

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1] range
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Reserve 20% of data for validation
)

# Create training data generator
train_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

# Create validation data generator
validation_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Print some details about the data
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {validation_generator.samples}")
print(f"Classes: {train_generator.class_indices}")
