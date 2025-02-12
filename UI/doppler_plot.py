import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import serial

class DopplerPlot():
    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.read_data = []
        self.plot_data = None
        self.save_csv = None
        self.save_img = None
        self.serial_connection = None

    def establish_serial_connection(self):
        self.serial_connection = serial.Serial(self.serial_port, baudrate=9600)
        print(f"Connected to Arduino on port {self.serial_port}")

    def read_serial_data(self, num_points):
        if self.serial_connection is None:
            print("Serial connection not established.")
            return

        for _ in range(num_points):
            try:
                data = self.serial_connection.readline().decode().strip()
                self.read_data.append(data)
            except UnicodeDecodeError:
                print("Invalid data received from Arduino")

    def plot_data(self):
        if not self.read_data:
            print("No data to plot.")
            return

        data_values = [list(map(float, line.split(',')) for line in self.read_data)]
        self.plot_data = pd.DataFrame(data_values, columns=["Value1", "Value2", "Value3","Value4"])

        # Plot the data
        plt.figure()
        self.plot_data.plot()
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.title("Arduino Data Plot")
        plt.grid(True)
        plt.show()

    def save_data_to_csv(self, filename):
        if self.plot_data is None:
            print("No data to save.")
            return

        self.plot_data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_plot_image(self, filename):
        if self.plot_data is None:
            print("No data to plot and save.")
            return

        plt.figure()
        self.plot_data.plot()
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.title("Arduino Data Plot")
        plt.grid(True)
        plt.savefig(filename)
        print(f"Plot image saved to {filename}")

    def close_serial_connection(self):
        if self.serial_connection:
            self.serial_connection.close()
            print("Serial connection closed")

if __name__ == "__main__":
    arduino_data = DopplerPlot(serial_port="/dev/ttyUSB0")
    arduino_data.establish_serial_connection()
    arduino_data.read_serial_data(num_points=100)
    arduino_data.plot_data()
    arduino_data.save_data_to_csv("arduino_data.csv")
    arduino_data.save_plot_image("arduino_plot.png")
    arduino_data.close_serial_connection()

