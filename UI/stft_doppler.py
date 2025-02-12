import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import pandas as pd

directory = "C:/Users/turan/OneDrive/Desktop/HGR_test/csv_files"
output_directory = "C:/Users/turan/OneDrive/Desktop/HGR_test/plots"

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

    #max_limit = 1000#my limit

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.title ('Sensör 1')
    plt.plot(data["Value1"], label="Node1", color="blue")
    #plt.ylim(0, max_limit)  # Set y-axis limit for subplot 1

    plt.subplot(2, 2, 2)
    plt.title ('Sensör 2')
    plt.plot(data["Value2"], label="Node2", color="red")
    #plt.ylim(0, max_limit)  # Set y-axis limit for subplot 1

    plt.subplot(2, 2, 3)
    plt.title('Sensör 3')
    plt.plot(data["Value3"], label="Node3", color="green")
    #plt.ylim(0, max_limit)  # Set y-axis limit for subplot 1

    plt.subplot(2, 2, 4)
    plt.title('Sensör 4')
    plt.plot(data["Value4"], label="Node4", color="brown")
    #plt.ylim(0, max_limit)  # Set y-axis limit for subplot 1

    plt.suptitle(file_name)
    #plt.legend()

    plt.savefig(f"{output_directory}/{file_name}.png")

    plt.close()

    data = None
    file_name = None

print("Succesfull")



