import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Assuming you have a directory containing images
image_directory = "D:/Desktop/handwrite_recognition/ml_project/figures2"
image_files = os.listdir(image_directory)

# Create a list to store images and labels
images = []
labels = []

# Load images and assign labels
for filename in image_files:
    image_path = os.path.join(image_directory, filename)
    img = load_img(image_path, target_size=(128, 128))  # Set your desired height and width
    img_array = img_to_array(img)
    images.append(img_array)
    labels.append(int(filename.split('_')[0]))

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values to be between 0 and 1
X_train /= 255.0
X_test /= 255.0
