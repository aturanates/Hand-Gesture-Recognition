import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow import _tf_uses_legacy_keras
from keras import layers
import matplotlib.pyplot as plt
import os

# Assuming you have your spectrogram images stored in a directory
spectrogram_directory = "D:/Desktop/handwrite_recognition/gunduz/spectogram"
height, width, channels = 480,640,4
# Load and preprocess the spectrogram images
def load_spectrogram_images(directory):
    spectrogram_images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Assuming your file names contain information about the class (label)
            #label = int(filename.split('_')[0])  # Extract label from file name
            label = filename.index(filename.split('_')[0])  # Extract label from file name

            img_path = os.path.join(directory, filename)

            # Load image, preprocess if necessary (e.g., normalization), and append to the list
            img = plt.imread(img_path)
            spectrogram_images.append(img)
            labels.append(label)

    return np.array(spectrogram_images), np.array(labels)

# Load spectrogram images and corresponding labels
spectrogram_images, labels = load_spectrogram_images(spectrogram_directory)

# Define the number of classes
num_classes = 3  # Replace with the actual number of classes

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
num_epochs = 20  # Adjust based on your observations of the training/validation performance

# Train the model
model.fit(spectrogram_images, labels, epochs=num_epochs, validation_split=0.2)

# Train the model
model.fit(spectrogram_images, labels, epochs=num_epochs, validation_split=0.2)
