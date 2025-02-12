import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import pandas as pd

directory = "D:/Desktop/handwrite_recognition/zeroK_turan_deneme/csv_files1"
output_directory = "D:/Desktop/handwrite_recognition/deneme2_png"

csv_files = glob.glob(os.path.join(directory, "*.csv"))

data = []

for csv_file in csv_files:
    file_name = os.path.split(csv_file)[-1]
    data = pd.read_csv(csv_file)
    #data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).astype(float))
    data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).apply(lambda y: float(y) if y.replace('.', '', 1).isdigit() else None))

    data = data.drop("Value", axis=1)

    #min_max_scaler = MinMaxScaler()
    #data[['Value1', 'Value2', 'Value3', 'Value4']] = min_max_scaler.fit_transform(data[['Value1', 'Value2', 'Value3', 'Value4']])

    plt.figure()
    #plt.subplot(2, 2, 1)
    plt.plot(data["Value1"], label="Node1", color="blue")
    plt.savefig(f"D:/Desktop/handwrite_recognition/zeroK_turan_deneme/raw_data2/{file_name}1.png")
    plt.close()

    #plt.subplot(2, 2, 2)
    plt.figure()
    plt.plot(data["Value2"], label="Node2", color="red")
    plt.savefig(f"D:/Desktop/handwrite_recognition/zeroK_turan_deneme/raw_data2/{file_name}2.png")
    plt.close()

    plt.figure()
    #plt.subplot(2, 2, 3)
    plt.plot(data["Value3"], label="Node3", color="green")
    plt.savefig(f"D:/Desktop/handwrite_recognition/zeroK_turan_deneme/raw_data2/{file_name}3.png")
    plt.close()

    plt.figure()
    #plt.subplot(2, 2, 4)
    plt.plot(data["Value4"], label="Node4", color="brown")
    plt.savefig(f"D:/Desktop/handwrite_recognition/zeroK_turan_deneme/raw_data2/{file_name}4.png")
    plt.close()

    #plt.suptitle(file_name)
    #plt.legend()

    #plt.savefig(f"D:/Desktop/handwrite_recognition/zeroK_turan_deneme/raw_data1/{file_name}.png")

    data = None
    file_name = None

print("Succesfull")



