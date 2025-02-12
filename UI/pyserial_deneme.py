import sys
import csv
import serial
import threading
import time
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QComboBox
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QPlainTextEdit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime
import glob
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import winsound
import plot_Raw
import plot_Spect

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

sensor_data = []

app = QApplication(sys.argv)

class DataSignal(QObject):
    update_data = pyqtSignal(list)

class ArduinoThread(threading.Thread):
    def __init__(self, data_signal):
        super(ArduinoThread, self).__init__()
        self.data_signal = data_signal
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            start_time = time.time()
            while not self._stop_event.is_set():
                try:
                    line = ser.readline().decode("utf-8").strip()
                    if line:
                        sensor_data.append(line)
                        self.data_signal.update_data.emit(sensor_data)
                    current_time = time.time()
                    if current_time-start_time >= 7:
                        self.stop()
                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                        #time.sleep(0.1)
                except:
                    pass

class SensorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Handwrite Recognition App')
        self.setGeometry(100, 100, 800, 600)

        self.directory = 'D:/Desktop/handwrite_recognition/data'

        self.label = str()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_data_collection)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_data_collection)
        layout.addWidget(self.stop_button)

        self.save_button = QPushButton('Save Data as CSV')
        self.save_button.clicked.connect(self.save_data_as_csv)
        layout.addWidget(self.save_button)

        self.plot_button = QPushButton('Plot Data')
        self.plot_button.clicked.connect(self.plot_data)
        layout.addWidget(self.plot_button)

        self.plot_spect_button = QPushButton('Plot Spectogram')
        self.plot_spect_button.clicked.connect(self.plot_spectogram)
        layout.addWidget(self.plot_spect_button)

        self.indir_text_edit = QTextEdit()
        self.indir_text_edit.setPlaceholderText("Input Directory")
        self.indir_text_edit.setMaximumHeight(30)
        layout.addWidget(self.indir_text_edit)

        self.outdir_text_edit = QTextEdit()
        self.outdir_text_edit.setPlaceholderText("Output Directory")
        self.outdir_text_edit.setMaximumHeight(30)
        layout.addWidget(self.outdir_text_edit)

        self.data_signal = DataSignal()
        self.data_signal.update_data.connect(self.update_data_display)

        self.data_display = QTextEdit()
        layout.addWidget(self.data_display)

        #self.plot_canvas = FigureCanvas(Figure())
        #layout.addWidget(self.plot_canvas)

        combo_layout = QVBoxLayout()
        """
        self.label1 = QLabel("Time")
        combo_layout.addWidget(self.label1)
        
        self.combo1 = QComboBox(self)
        self.combo1.addItem("3sn")
        self.combo1.addItem("6sn")
        self.combo1.addItem("9sn")
        self.combo1.activated[str].connect(self.process_time)
        combo_layout.addWidget(self.combo1)

        self.label2 = QLabel("Time Duration")
        combo_layout.addWidget(self.label2)
        
        self.combo2 = QComboBox(self)
        self.combo2.addItem("0.01")
        self.combo2.addItem("0.005")
        self.combo2.addItem("0.001")
        self.combo2.activated[str].connect(self.process_time_duration)
        combo_layout.addWidget(self.combo2)
        """
        #combo_layout.addStretch()

        layout.addLayout(combo_layout)

        self.label_text_edit = QTextEdit()
        self.label_text_edit.setPlaceholderText("Label:")
        self.label_text_edit.setMaximumHeight(30)
        layout.addWidget(self.label_text_edit)

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.train_ml)
        combo_layout.addWidget(self.train_button)

        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(self.test_ml)
        combo_layout.addWidget(self.test_button)

        port_text_edit = QTextEdit()
        port_text_edit.setPlaceholderText("Port:")
        port_text_edit.setMaximumHeight(30)
        layout.addWidget(port_text_edit)

        self.central_widget.setLayout(layout)

        self.arduino_thread = None

    def start_data_collection(self):
        self.arduino_thread = ArduinoThread(self.data_signal)
        self.arduino_thread.start()
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

    def stop_data_collection(self):
        if self.arduino_thread:
            self.arduino_thread.stop()
            self.arduino_thread.join()
            self.arduino_thread = None
            sensor_data.clear()
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
    def save_data_as_csv(self):
        if sensor_data:
            label = self.label_text_edit.toPlainText()
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"C:/Users/turan/OneDrive/Desktop/HGR_test/csv_files/{label}_{current_time}.csv"

            with open(filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                #csv_writer.writerow(["Time","Value"])
                csv_writer.writerow(["Value"])

                current_milliseconds = int(round(time.time() * 1000))

                for data in sensor_data:
                    csv_writer.writerow([data])
                    current_milliseconds += 1

                #csv_writer.writerow(sensor_data)

    def data_framing(self):
        csv_files = glob.glob(os.path.join(self.directory, '*.csv'))
        date_formats = ["%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M-%S"]
        dates = []

        for csv_file in csv_files:
            for date_format in date_formats:
                try:
                    date = pd.to_datetime(os.path.splitext(os.path.basename(csv_file))[0], format=date_format)
                    dates.append((date, csv_file))
                    break
                except ValueError:
                    pass

    def plot_data(self):
        input_dir = self.indir_text_edit.toPlainText()
        output_dir = self.outdir_text_edit.toPlainText()
        plot_Raw.DataPlotter(directory=input_dir, output_directory=output_dir)

    def plot_spectogram(self):
        input_dir = self.indir_text_edit.toPlainText()
        output_dir = self.outdir_text_edit.toPlainText()
        plot_Raw.DataPlotter(directory=input_dir, output_directory=output_dir)


    def update_data_display(self, data):
        if data:
            data_str = '\n'.join(data)
            self.data_display.setPlainText(data_str)
            """
            # Ensure autoscroll
            cursor = self.data_display.textCursor()
            cursor.movePosition(QPlainTextEdit.End)
            self.data_display.setTextCursor(cursor)
            self.data_display.ensureCursorVisible()
            """
    def process_time(self):
        pass
    def process_time_duration(self):
        pass
    def train_ml(self):
        pass
    def test_ml(self):
        pass
if __name__ == '__main__':

    window = SensorApp()
    window.show()
    sys.exit(app.exec_())