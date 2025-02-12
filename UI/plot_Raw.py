import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class DataPlotter:
    def __init__(self, directory, output_directory):
        self.directory = "D:/Desktop/handwrite_recognition/" + directory
        self.output_directory = "D:/Desktop/handwrite_recognition/" + output_directory

    def process_csv_files(self):
        csv_files = glob.glob(os.path.join(self.directory, "*.csv"))

        for csv_file in csv_files:
            file_name = os.path.split(csv_file)[-1]
            data = pd.read_csv(csv_file)
            data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).astype(float))
            data[['Value1', 'Value2', 'Value3', 'Value4']] = data['Value'].apply(lambda x: pd.Series(str(x).split(',')).apply(lambda y: float(y) if y.replace('.', '', 1).isdigit() else None))
            data = data.drop("Value", axis=1)

            self.plot_and_save(data, file_name)

            data = None
            file_name = None

        print("Successful")

    def plot_and_save(self, data, file_name):
        plt.figure()

        plt.subplot(2, 2, 1)
        plt.plot(data["Value1"], label="Node1", color="blue")

        plt.subplot(2, 2, 2)
        plt.plot(data["Value2"], label="Node2", color="red")

        plt.subplot(2, 2, 3)
        plt.plot(data["Value3"], label="Node3", color="green")

        plt.subplot(2, 2, 4)
        plt.plot(data["Value4"], label="Node4", color="brown")

        plt.suptitle(file_name)
        # plt.legend()

        output_path = os.path.join(self.output_directory, f"{file_name}.png")
        plt.savefig(output_path)

        plt.close()