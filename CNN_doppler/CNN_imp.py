import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import glob
import pandas as pd

# Assuming you have your sensor data in a NumPy array with shape (num_samples, num_sensors)
# Here, I'm generating some random data for demonstration purposes.
nperseg = 16
noverlap = nperseg/2
num_samples = 246
num_sensors = 4
sampling_frequency = 246/5

directory = "D:/Desktop/handwrite_recognition/turan/csv_files"
output_directory = "D:/Desktop/handwrite_recognition/deneme2_png"

csv_files = glob.glob(os.path.join(directory, "*.csv"))

data = []

for csv_file in csv_files:
    file_name = os.path.split(csv_file)[-1]
    data = pd.read_csv(csv_file)
    #data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).astype(float))
    data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).apply(lambda y: float(y) if y.replace('.', '', 1).isdigit() else None))

    data = data.drop("Value", axis=1)

    min_max_scaler = MinMaxScaler()
    data[['Value1', 'Value2', 'Value3', 'Value4']] = min_max_scaler.fit_transform(data[['Value1', 'Value2', 'Value3', 'Value4']])

    signal1 = data["Value1"].to_numpy()
    signal2 = data["Value2"].to_numpy()
    signal3 = data["Value3"].to_numpy()
    signal4 = data["Value4"].to_numpy()

    signal1 = signal1 - np.mean(signal1, axis=-1)
    signal2 = signal2 - np.mean(signal2, axis=-1)
    signal3 = signal3 - np.mean(signal3, axis=-1)
    signal4 = signal4 - np.mean(signal4, axis=-1)

    _, _, Zxx1 = signal.stft(signal1, sampling_frequency, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx2 = signal.stft(signal2, sampling_frequency, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx3 = signal.stft(signal3, sampling_frequency, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx4 = signal.stft(signal4, sampling_frequency, nperseg=nperseg, noverlap=noverlap)

    Zxx1 = abs(Zxx1)
    Zxx2 = abs(Zxx2)
    Zxx3 = abs(Zxx3)
    Zxx4 = abs(Zxx4)

    spectogram1 =
"""
# Generate random sensor data
sensor_data = np.random.rand(num_samples, num_sensors)


# Function to compute STFT spectrograms
def compute_spectrograms(data, sampling_frequency):
    _, _, Zxx = stft(data, fs=sampling_frequency, nperseg=256)
    return np.abs(Zxx)

# Compute STFT spectrograms for each sensor
spectrograms = np.zeros((num_samples, num_sensors, 129, num_samples // 2 + 1))  # Assuming nperseg=256

for i in range(num_sensors):
    spectrograms[:, i, :, :] = compute_spectrograms(sensor_data[:, i], sampling_frequency)
"""


# Reshape the spectrograms to be suitable for CNN input
spectrograms = spectrograms.reshape((num_samples, num_sensors, 129, num_samples // 2 + 1, 1))

num_classes = 10  # Replace 10 with the actual number of classes in your classification task

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(num_sensors, 129, num_samples // 2 + 1, 1)),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (you'll need labels for training)
# model.fit(x=spectrograms, y=labels, epochs=num_epochs)

# You can replace the layers and parameters in the CNN model with what fits your specific problem.
