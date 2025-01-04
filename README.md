### README for Gesture Recognition Project

# Smart TV Gesture Recognition

This project aims to develop a gesture recognition system for a smart TV using deep learning models, allowing users to control the TV via specific hand gestures detected through a webcam. The goal is to create a system that can recognize five distinct gestures and map them to TV commands. This repository contains the code and instructions to train, evaluate, and deploy a gesture recognition model.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Model Training](#model-training)
- [Setup and Installation](#setup-and-installation)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Problem Statement

As a data scientist at a home electronics company developing a feature for a smart TV that can recognize user gestures and control the TV without the need for a remote. The five gestures to be recognized are:

1. **Thumbs up**: Increase the volume
2. **Thumbs down**: Decrease the volume
3. **Left swipe**: Jump backward 10 seconds
4. **Right swipe**: Jump forward 10 seconds
5. **Stop**: Pause the movie

Each gesture is captured in short video sequences, where each video consists of 30 frames (or images). The challenge is to build a deep learning model that can accurately classify these gestures.

## Dataset

The dataset consists of videos of 5 different gestures, each labeled with a numeric value (0-4). These videos are split into training and validation sets.

- **Train Folder**: Contains training videos for gesture recognition.
- **Val Folder**: Contains validation videos for testing the model’s performance.
- **Each Video**: 
  - 30 frames (images) representing one video.
  - Two possible image dimensions: 360x360 or 120x160.

The dataset is provided in CSV format with columns:
- Video folder name (which contains the 30 images for the video)
- Gesture name
- Numeric gesture label (0-4)

## Architecture

### 1. **CNN + RNN Stack**

This approach involves using Convolutional Neural Networks (CNNs) followed by Recurrent Neural Networks (RNNs) for gesture classification.

- **CNN**: Each frame in a video is passed through a CNN to extract spatial features.
- **RNN**: The sequence of features is passed to an RNN (GRU or LSTM) to capture the temporal dependencies between frames in the video.

### 2. **3D Convolutional Network (Conv3D)**

Instead of processing each frame separately, this method uses 3D convolutions to directly process the entire video sequence as a 3D tensor. The network learns both spatial and temporal patterns simultaneously.

- **Conv3D**: Operates on a sequence of frames (video), extending traditional 2D convolutions to 3D.
- **Output**: After 3D convolution, the model outputs predictions for the gesture classification.

## Model Training

### Requirements

- **Python 3.x**
- **TensorFlow 2.x** (for Keras and model training)
- **NumPy**
- **OpenCV** (for video frame preprocessing)
- **Matplotlib** (for plotting training metrics)

### Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/gesture-recognition.git](https://github.com/RavishankarDuMCA10/Gesture_Recognition)
   cd gesture-recognition
   ```

### Training the Model

To train the model, use Jupyter Notebook or Colab:

- `--model_type`: Choose between `cnn_rnn` or `conv3d` to train with different architectures.
- `--batch_size`: Set the batch size for training.
- `--epochs`: Set the number of epochs.

The model will be trained and evaluated on the validation set. The best model will be saved in the `models/` folder.

## Setup and Installation

### Prerequisites

- Python 3.x (preferably 3.6+)
- A compatible deep learning environment (TensorFlow 2.x)

## File Structure

```plaintext
gesture-recognition/
│
├── data/                          # Folder for dataset
│   ├── train/                     # Training set
│   └── val/                       # Validation set
│
├── models/                        # Folder to save trained models
│   └── best_model.h5              # The best model after training
│
├── scripts/                       # Python scripts
│   └── Gesture_Recognition.ipynb  # Training/Validation and testing script
│   
├── README.md                      # Project documentation
└── LICENSE                        # Project license
```

## Acknowledgments

- **Dataset Source**: The dataset used for this project contains hand gesture videos and was created to help build gesture recognition systems.
- **Keras and TensorFlow**: The project uses TensorFlow/Keras for model training and evaluation.
- **Additional Libraries**: OpenCV for video frame processing and NumPy for numerical operations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides an overview of the project, installation instructions, and guides users to train, evaluate, and deploy the gesture recognition model.
