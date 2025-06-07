# Sign Language Converter

A real-time sign language recognition system that converts sign language gestures into text using computer vision and deep learning. This project implements the recognition system using a pre-trained model with 99.2% accuracy.

## Overview

This project implements a real-time sign language recognition system that can detect and interpret sign language gestures using a webcam. The system uses computer vision techniques to capture hand movements and a pre-trained deep learning model (99.2% accuracy) to classify the signs into text. The GUI interface allows users to combine recognized characters into words and provides word suggestions based on the input.

## Dataset

The model was trained on a comprehensive dataset of sign language gestures:
- Contains 26 classes (A-Z) of sign language gestures
- Each class contains multiple samples for robust training
- Dataset is split into:
  - 80% for training
  - 20% for validation
- Organized in a structured directory format for easy access and processing

## Features

- Real-time sign language detection using webcam
- Character-by-character sign recognition
- GUI interface for word formation
- Word suggestions based on recognized characters
- High accuracy recognition (99.2% on pre-trained model)
- Pre-trained model ready to use

## Prerequisites

- Python 3.8 or higher
- Webcam
- CUDA-compatible GPU (recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sign-Language-Converter.git
cd Sign-Language-Converter
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
   - Download the model from [Google Drive](https://drive.google.com/file/d/1pM4blN8rhbX5kWMbGGQaEuxnpW8YX8JY/view?usp=sharing)
   - Place the model file in the project root directory

## Usage

### Real-time Prediction

To run the real-time sign language recognition:

```bash
python realtime_predict.py
```

### GUI Interface

To launch the GUI version with word formation capabilities:

```bash
python gui_test.py
```

The GUI provides the following features:
- Real-time character recognition from sign language
- Character input for word formation
- Word suggestions based on recognized characters
- Easy word building interface

### Testing the Model

To test the model's performance:

```bash
python test_model.py
```

## Project Structure

- `realtime_predict.py`: Main script for real-time sign language recognition
- `gui_test.py`: GUI interface for the sign language converter with word formation capabilities
- `test_model.py`: Script for testing the model
- `requirements.txt`: List of required Python packages
- `SignLanguageConverter_Presentation.pdf`: Project documentation and presentation
- `data/`: Directory containing the dataset
  - `train/`: Training data (80% of dataset)
  - `validation/`: Validation data (20% of dataset)
  - `A-Z/`: Individual gesture class directories

## Dependencies

- numpy==1.24.3
- tensorflow==2.14.0
- opencv-python==4.8.0.76
- Pillow==10.0.0
- matplotlib==3.7.2
- scikit-learn==1.3.0
- keras==2.14.0
- mediapipe==0.10.21
- cvzone==1.6.1

## How It Works

1. The system captures video feed from the webcam
2. Hand landmarks are detected using MediaPipe
3. The detected hand gestures are processed and fed into the pre-trained model
4. The model predicts the sign language gesture and converts it to a character
5. The recognized character is displayed on screen
6. Users can combine characters to form words using the GUI
7. The system provides word suggestions based on the recognized characters

## Model Information

The project uses a pre-trained model with the following specifications:
- Accuracy: 99.2%
- Pre-trained and ready to use
- Optimized for real-time recognition
- No additional training required
- Trained on a comprehensive dataset of sign language gestures
- Achieved high accuracy through 80-20 training-validation split

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand landmark detection
- TensorFlow and Keras for deep learning framework
- OpenCV for computer vision capabilities