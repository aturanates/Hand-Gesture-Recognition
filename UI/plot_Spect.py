import os
import glob
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class SpectrogramProcessor:
    def __init__(self, directory, output_directory, nperseg=16, noverlap_ratio=0.5, fs=250/5):
        self.directory = directory
        self.output_directory = output_directory
        self.nperseg = nperseg
        self.noverlap = int(nperseg * noverlap_ratio)
        self.fs = fs

    def process_csv_files(self):
        csv_files = glob.glob(os.path.join(self.directory, "*.csv"))

        for csv_file in csv_files:
            file_name = os.path.split(csv_file)[-1]
            data = pd.read_csv(csv_file)
            data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).apply(lambda y: float(y) if y.replace('.', '', 1).isdigit() else None))
            data = data.drop("Value", axis=1)

            min_max_scaler = MinMaxScaler()
            data[['Value1', 'Value2', 'Value3', 'Value4']] = min_max_scaler.fit_transform(data[['Value1', 'Value2', 'Value3', 'Value4']])

            for i in range(1, 5):
                signal_data = data[f"Value{i}"].to_numpy()
                signal_data = signal_data - np.mean(signal_data, axis=-1)
                f, t, Zxx = signal.stft(signal_data, self.fs, nperseg=self.nperseg, noverlap=self.noverlap)

                plt.figure()
                plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=signal_data.max(), shading='gouraud')
                plt.savefig(os.path.join(self.output_directory, f"{file_name}{i}.png"))
                plt.close()

            data = None
            file_name = None

        print("Successful")