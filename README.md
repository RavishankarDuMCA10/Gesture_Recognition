# Smart TV Gesture Recognition

This project aims to develop a gesture recognition system for a smart TV using deep learning models, allowing users to control the TV via specific hand gestures detected through a webcam. The goal is to create a system that can recognize five distinct gestures and map them to TV commands. This repository contains the code and instructions to train, evaluate, and deploy a gesture recognition model.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture](#architecture)
  - [3D Convolutional Network (Conv3D)](#3d-convolutional-network-conv3d)
  - [CNN + RNN Stack](#cnn--rnn-stack)
  - [Transfer Learning](#transfer-learning)
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

### 1. **3D Convolutional Network (Conv3D)**

Instead of processing each frame separately, this method uses 3D convolutions to directly process the entire video sequence as a 3D tensor. The network learns both spatial and temporal patterns simultaneously.

- **Conv3D**: Operates on a sequence of frames (video), extending traditional 2D convolutions to 3D.
- **Output**: After 3D convolution, the model outputs predictions for the gesture classification.

### 2. **CNN + RNN Stack**

This approach involves using Convolutional Neural Networks (CNNs) followed by Recurrent Neural Networks (RNNs) for gesture classification.

- **CNN**: Each frame in a video is passed through a CNN to extract spatial features.
- **RNN**: The sequence of features is passed to an RNN (GRU or LSTM) to capture the temporal dependencies between frames in the video.


### 3. **Transfer Learning**

Transfer learning is used in this project to leverage pre-trained models on ImageNet, allowing the model to benefit from features learned from a much larger and diverse dataset. This helps reduce training time and improves performance, especially in cases with limited data.

- **Why Transfer Learning?** Transfer learning involves fine-tuning a pre-trained model (such as **ResNet**, **VGG16**, or **InceptionV3**) that has been trained on ImageNet, and then adapting it for the gesture recognition task.
- **How it works**: 
  - **Pretrained CNN**: A CNN like ResNet50 is pre-trained on ImageNet for image classification tasks.
  - **Fine-Tuning**: The pre-trained CNN is adapted for the gesture recognition task by replacing the final layers with new ones that correspond to our gesture classes.
  - **Freezing Layers**: Initially, most of the layers of the pre-trained model are frozen (i.e., they aren't updated during training), and only the final layers are trained. As the training progresses, some earlier layers may also be fine-tuned to adapt to the new data.
<!---
#### Implementation of Transfer Learning:
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the pre-trained ResNet50 model, excluding the final dense layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pooling to reduce the spatial dimensions
x = Dense(1024, activation='relu')(x)
x = Dense(5, activation='softmax')(x)  # 5 classes for gesture recognition

# Define the full model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
-->
**Benefits of Transfer Learning:**
- Reduces the time to train a model.
- Improves performance with smaller datasets by leveraging knowledge learned from a larger, general dataset.
- Enables the use of sophisticated models like ResNet and InceptionV3 without requiring vast computational resources.
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
   git clone [https://github.com/RavishankarDuMCA10/gesture-recognition.git]
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
