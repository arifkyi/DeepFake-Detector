Deepfake Detection Project

This repository contains scripts and instructions for training a deepfake detection model and running inference on videos to detect deepfakes.
Directory Structure

Organize your data and scripts as follows:

bash

/data
├── authentic_videos
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── deepfake_videos
│   ├── video3.mp4
│   ├── video4.mp4
│   └── ...
├── authentic_frames
│   ├── frame1.jpg
│   ├── frame2.jpg
│   └── ...
└── deepfake_frames
    ├── frame1.jpg
    ├── frame2.jpg
    └── ...
/model
└── deepfake_detection_model.h5
/scripts
├── extract_frames.py
├── preprocess_data.py
└── train_model.py

Setup Instructions
Step 1: Create a Virtual Environment and Install Dependencies

    Create and activate a virtual environment (optional but recommended):

    bash

python3 -m venv deepfake_env
source deepfake_env/bin/activate

Upgrade pip:

bash

pip install --upgrade pip

Create a requirements.txt file with the following content:

text

numpy
opencv-python
tensorflow

Install required packages:

bash

    pip install -r requirements.txt

Step 2: Extract Frames from Videos

Create a script named extract_frames.py to extract frames from videos:

python

import cv2
import os

def video_to_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        frame_number += 1
    
    cap.release()
    print("Finished extracting frames.")

if __name__ == "__main__":
    video_path = 'your_video.mp4'  # Path to your video file
    output_folder = 'output_frames'  # Folder to save the extracted frames
    video_to_frames(video_path, output_folder)

Step 3: Preprocess Data

Create a script named preprocess_data.py to preprocess the images:

python

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path, target_size=(299, 299)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

def load_data(directory, label):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = preprocess_image(img_path)
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

def preprocess_data(authentic_dir, deepfake_dir):
    authentic_data, authentic_labels = load_data(authentic_dir, label=0)
    deepfake_data, deepfake_labels = load_data(deepfake_dir, label=1)

    X = np.concatenate((authentic_data, deepfake_data), axis=0)
    y = np.concatenate((authentic_labels, deepfake_labels), axis=0)

    y = to_categorical(y, num_classes=2)

    return X, y

if __name__ == "__main__":
    authentic_dir = "data/authentic_frames"
    deepfake_dir = "data/deepfake_frames"

    X, y = preprocess_data(authentic_dir, deepfake_dir)

    np.save("preprocessed_data/X.npy", X)
    np.save("preprocessed_data/y.npy", y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

Step 4: Train the Model

Create a script named train_model.py to train the deepfake detection model:

python

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model

def train_model(X_train, y_train, X_val, y_val):
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return model

if __name__ == "__main__":
    X = np.load("preprocessed_data/X.npy")
    y = np.load("preprocessed_data/y.npy")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, X_val, y_val)

    model.save("model/deepfake_detection_model.h5")

Step 5: Run Inference

Finally, use your existing inference script (e.g., detect_deepfake_video.py) to run inference using the trained model:

python

python detect_deepfake_video.py video_path model/deepfake_detection_model.h5 model/

Copy the content into your GitHub repository's README file. This should provide clear instructions on how to set up and run the project. Let me know if you need any further assistance!
ChatGPT can make mistakes. Check important info.
