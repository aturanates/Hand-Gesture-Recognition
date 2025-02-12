import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

data = pd.read_csv("data/measure1_2023-11-02_17-05-00.csv")

data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).astype(float))

data = data.drop("Value",axis=1)

min_max_scaler = MinMaxScaler()
data[['Value1', 'Value2', 'Value3', 'Value4']] = min_max_scaler.fit_transform(data[['Value1', 'Value2', 'Value3', 'Value4']])



#plt.plot(data["Value1"],label="Node1",color = "blue")
#plt.plot(data["Value2"],label="Node2",color = "red")
#plt.plot(data["Value3"],label="Node3",color = "green")
#plt.plot(data["Value4"],label="Node4",color = "purple")

"""
plt.subplot(2, 1, 1)
plt.plot(data["Value1"],label="Node1",color = "blue")

plt.subplot(2, 2, 2)
plt.plot(data["Value2"],label="Node2",color = "red")

plt.subplot(2, 2, 3)
plt.plot(data["Value3"],label="Node3",color = "green")

plt.subplot(2, 2, 4)
plt.plot(data["Value4"],label="Node4",color = "brown")

plt.show()
"""

fs = 24e9

nperseg = 16
noverlap = 8

frequencies, times, Sxx = spectrogram(data["Value1"], fs=fs, nperseg=nperseg, noverlap=noverlap)

plt.pcolormesh(times, frequencies / 1e9, 10 * np.log10(Sxx), shading='auto')

plt.colorbar(label='dB')
plt.ylabel('Frequency (GHz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram')
plt.show()