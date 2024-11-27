# Pytorch-CNN-Classifier 🌐
A CNN-based deep learning project for recognizing handwritten digits with data augmentation and validation metrics.

## Credits 🤖
[![Building a Neural Network with PyTorch in 15 Minutes | Coding Challenge](https://img.youtube.com/vi/mozBidd58VQ&list=LL/0.jpg)](https://www.youtube.com/watch?v=mozBidd58VQ&list=LL) - 
**Nicholas Renotte**.
The base code for this project was adapted from Nicholas Renotte. While the original concept and code were used as a foundation, several modifications were made to suit the specific functionality and features of this classifier.

## Demo 🎬



## The Code files: 📄
**torchnn.py:** includes a convolutional neural network for MNIST digit classification with data augmentation, training, validation, model saving/loading, and inference capabilities using PyTorch.

## Functionality ⚙️
**Data Preparation and Augmentation:** The code sets up data loaders for the MNIST dataset, with separate transformations for training (including data augmentation like rotation and flipping) and testing. This improves the model's generalization by introducing variability in the training data.

**Model Definition:** A convolutional neural network (CNN) is defined using PyTorch's nn.Sequential. The architecture includes multiple convolutional layers, activation functions, and a final fully connected layer for classifying MNIST digits into 10 classes.

**Training and Validation Workflow:**

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

## Theory 🧮

## the Whys:❓
