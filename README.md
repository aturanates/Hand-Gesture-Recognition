# Hand Gesture Recognition Using Micro-Doppler Signatures


A novel hand gesture recognition system utilizing micro-Doppler signatures from 24.125 GHz Doppler radar sensors. This project demonstrates high-accuracy gesture recognition using raw Doppler signals processed through advanced machine learning algorithms.

## Project Overview

This system uses four strategically placed 24.125 GHz microwave Doppler sensors to capture hand movements and recognize written letters through their unique Doppler signatures. The project implements various machine learning and deep learning approaches to achieve high recognition accuracy.

## System Architecture

![System Architecture](./images/system-architecture.svg)

The system architecture consists of multiple layers:
- **Data Collection Layer**: Custom PyQt5 interface for capturing and managing sensor data
- **Hardware Layer**: Arduino Mega controller interfacing with four 24.125 GHz Doppler sensors
- **Signal Processing Layer**: STFT and spectogram generation for feature extraction
- **Machine Learning Layer**: Multiple classification approaches including traditional ML, CNN, and TSMixer models

### Key Features

- Real-time hand gesture recognition using Doppler radar sensors
- Multi-sensor configuration for comprehensive motion capture
- Advanced signal processing using Short-Time Fourier Transform (STFT)
- Multiple classification approaches:
  - Traditional ML (KNN, SVM, Random Forest)
  - Deep Learning (CNN with both single and multi-channel architectures)
  - Time Series Analysis (TSMixer)
- Custom-built data collection and processing pipeline
- High accuracy rates (up to 99% with multi-channel CNN)

## Technical Details

### Hardware Components

- 4x 24.125 GHz Microwave Doppler Sensors
- Arduino Mega for data acquisition
- Custom-designed amplification circuits
- 3D printed sensor mounting frame

### Software Stack

- Python for data processing and model implementation
- PyQt5 for the data collection interface
- TensorFlow/Keras for deep learning models
- Scikit-learn for traditional ML algorithms
- SciPy for signal processing

## Results

The system achieved impressive recognition rates across different classification approaches:

| Algorithm | Test Accuracy |
|-----------|---------------|
| KNN | 100% |
| SVM | 98% |
| Random Forest | 98% |
| CNN (Single Channel) | 97% |
| CNN (Multi Channel) | 99% |
| TSMixer | 97% |

## Directory Structure

```
├── data_collection/
│   ├── arduino/
│   └── interface/
├── preprocessing/
│   ├── stft/
│   └── signal_processing/
├── models/
│   ├── traditional_ml/
│   ├── cnn/
│   └── tsmixer/
└── evaluation/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Arduino environment:
- Install Arduino IDE
- Upload the provided sketch to Arduino Mega

## Usage

1. Start the data collection interface:
```bash
python interface/main.py
```

2. Run the training pipeline:
```bash
python train.py --model [cnn/tsmixer/traditional]
```

3. Evaluate models:
```bash
python evaluate.py --model [model_name]
```

## Publications

Research from this project has been presented in the academic paper: "Hand Gesture Recognition Using Micro-Doppler Signatures" (2024).

## Author

- Ahmet Turan Ateş


## Acknowledgments

- Gebze Technical University for providing research facilities
- Open-source community for various tools and libraries used in this project