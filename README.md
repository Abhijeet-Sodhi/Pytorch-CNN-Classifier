# Pytorch-CNN-Classifier 🌐
The torchnn.py project is based on a Convolutional Neural Network (CNN) that is designed to classify images from the MNIST dataset, which contains grayscale images of handwritten digits (0–9).

## Credits 🤖
[![Building a Neural Network with PyTorch in 15 Minutes | Coding Challenge](https://img.youtube.com/vi/mozBidd58VQ&list=LL/0.jpg)](https://www.youtube.com/watch?v=mozBidd58VQ&list=LL) - 
**Nicholas Renotte**.
The base code for this project was adapted from Nicholas Renotte. While the original concept and code were used as a foundation, several modifications were made to suit the specific functionality and features of this classifier.

## Demo 🎬
**Training**

https://github.com/user-attachments/assets/ef703717-7e6c-4e3e-81ee-e2a76d17e080

**Testing**

https://github.com/user-attachments/assets/fa3886b0-40dd-4055-a232-d461972e35a7

## The Code files: 📄
**torchnn.py:** includes a convolutional neural network for MNIST digit classification with data augmentation, training, validation, model saving/loading, and inference capabilities using PyTorch.

## Functionality ⚙️
**Data Preparation and Augmentation:** The code sets up data loaders for the MNIST dataset, with separate transformations for training (including data augmentation like rotation and flipping) and testing. This improves the model's generalization by introducing variability in the training data.

**Model Definition:** A convolutional neural network (CNN) is defined using PyTorch's nn.Sequential. The architecture includes multiple convolutional layers, activation functions, and a final fully connected layer for classifying MNIST digits into 10 classes.

**Epoch Management:** The training loop iteratively adjusts model weights for 10 epochs.

**Metrics:** Training loss and accuracy are computed in train_epoch, while validation metrics are computed in validate_epoch.

**Best Model Saving:** The model with the lowest validation loss is saved as best_model.pt.

**Inference:** The code allows users to make predictions on new images. The image is preprocessed, and the trained model predicts its class, enabling inference capabilities outside the training loop.

**Command-Line Interface (CLI):** Users can choose between training the model **(--train)** or running inference on a specific image **(--infer <image_path>)** using CLI arguments.

**Modular Design:** Key functionalities like training, validation, model saving/loading, and inference are modularized into functions, improving readability and reusability.

**Error Handling:** The code includes basic error handling for loading the model, ensuring robustness during inference.

## Installation 💻
To run the torchnn.py MNIST classifier project, you'll need to install the following dependencies:

*pip install torch==2.5.1*    -  download from here: **https://pytorch.org/**

*pip install torchvision==0.20.1*  

*pip install numpy==2.0.2*  

*pip install Pillow==10.2.0*  

## What is a Neural Network? ❓
A **neural network** is a machine learning model inspired by the structure of the human brain. It consists of interconnected layers:

![image](https://github.com/user-attachments/assets/23c578dc-c1d6-43b5-93b5-c9dc70c9f187)

**Input Layer:** Takes the raw data (in this case, the pixel values of MNIST images).

**Hidden Layers:** Perform computations by using neurons that apply weights to their inputs, sum them up, add biases, and pass the result through an activation function.

**Output Layer:** Provides predictions, which in this project is the probability distribution over 10 classes (digits 0–9).

**Neuron:** A node that holds a value.

**Activation:** A measure of a neuron’s output after applying weights, biases, and an activation function.

**Activation Function:** Introduces non-linearity to help the network learn complex patterns. Common functions include ReLU, Sigmoid, and Softmax.

The MNIST classifier uses **feedforward neural networks** where the output of one layer serves as the input for the next, following a forward propagation process.

## Types of Neural Networks 📈
**Feedforward Neural Networks:** Data flows in one direction, from input to output.

**Convolutional Neural Networks (CNNs):** Specialized for image data; these include layers like convolution, pooling, and fully connected layers for feature extraction and classification.

**Recurrent Neural Networks (RNNs):** Designed for sequence data with feedback loops for tasks like time-series prediction.

The project uses a **CNN** because of its efficiency in recognizing patterns in image data.

## How CNNs Work in This Code 🔧
Convolutional Neural Networks (CNNs) are designed to process image data by extracting features through several layers:

![image](https://github.com/user-attachments/assets/f67a9b89-df4d-4fd4-b8c6-3ab4fa24e3d3)

**Convolutional Layers:**
These layers apply filters (small matrices) to the image, extracting features like edges, textures, and patterns. In ImageClassifier, the convolutional layers are defined as:

*nn.Conv2d(1, 32, (3,3)):* Extracts 32 feature maps from the grayscale input image.

*nn.Conv2d(32, 64, (3,3)):* Builds on the extracted features, producing 64 feature maps.

*nn.Conv2d(64, 64, (3,3)):* Further deepens feature extraction with another set of 64 feature maps.

**Activation Function:**
After each convolutional step, a non-linearity (typically ReLU: *nn.ReLU()*) is applied to help the network learn complex patterns and relationships within the data.

**Adaptive Pooling:**
The *nn.AdaptiveAvgPool2d((1, 1))* layer dynamically reduces the dimensions of the feature maps, retaining only the most critical information from each map while minimizing computational overhead.

**Flattening and Fully Connected Layer:**
After pooling, the feature maps are flattened using *nn.Flatten()* and passed through a fully connected layer (*nn.Linear(64, 10)*), mapping the extracted features to 10 output classes corresponding to the digits (0–9).

## the Data: 📊
The training data is augmented using transformations like **RandomRotation** and **RandomHorizontalFlip**. These increase the diversity of the dataset by randomly altering images, making the model more robust to variations.

**Validation Metrics**
The code computes the following during training:

**Loss:** Measures how far the model's predictions are from the true labels.

**Accuracy:** Calculates the proportion of correctly predicted examples. These metrics are displayed for both training and validation datasets to monitor progress.
