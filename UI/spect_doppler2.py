import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import pandas as pd

nperseg = 16
noverlap = nperseg/2
fs = 250/5

directory = "C:/Users/turan/OneDrive/Desktop/HGR_data_old/project_data/csv_files"
output_directory = "C:/Users/turan/OneDrive/Desktop/HGR_data_old/project_data/spectograms"

csv_files = glob.glob(os.path.join(directory, "*.csv"))

data = []

for csv_file in csv_files:

    file_name = os.path.split(csv_file)[-1]
    data = pd.read_csv(csv_file)

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

    f1, t1, Zxx1 = signal.stft(signal1, fs, nperseg=nperseg, noverlap=noverlap)
    f2, t2, Zxx2 = signal.stft(signal2, fs, nperseg=nperseg, noverlap=noverlap)
    f3, t3, Zxx3 = signal.stft(signal3, fs, nperseg=nperseg, noverlap=noverlap)
    f4, t4, Zxx4 = signal.stft(signal4, fs, nperseg=nperseg, noverlap=noverlap)


    plt.figure()

    plt.subplot(2, 2, 1)
    plt.pcolormesh(t1, f1, np.abs(Zxx1), vmin=0, vmax=signal1.max(), shading='gouraud')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.pcolormesh(t2, f2, np.abs(Zxx2), vmin=0, vmax=signal2.max(), shading='gouraud')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.pcolormesh(t3, f3, np.abs(Zxx3), vmin=0, vmax=signal3.max(), shading='gouraud')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.pcolormesh(t4, f4, np.abs(Zxx4), vmin=0, vmax=signal4.max(), shading='gouraud')
    plt.axis('off')

    plt.savefig(f"C:/Users/turan/OneDrive/Desktop/HGR_data_old/data_all/spectograms/{file_name}.png")

    plt.close()

    data = None
    file_name = None
