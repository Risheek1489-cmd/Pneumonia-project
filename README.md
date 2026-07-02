 # Pneumonia X-Ray Classifier
An AI model that detects pneumonia from chest X-ray images. Built by retraining ResNet-18 on a chest X-ray dataset using an NVIDIA Jetson Orin Nano. Given an X-ray image, the model classifies it as NORMAL or PNEUMONIA along with a confidence score. ![Pneumonia detection test output](direct image link here)
 # The Algorithm
This project uses transfer learning on ResNet-18, a convolutional neural network (CNN) that is 18 layers deep. Instead of training from scratch, the model starts with weights pre-trained on ImageNet, and the final fully-connected layer is replaced and retrained to classify two classes: normal lungs and pneumonia. Pneumonia shows up on chest X-rays as areas of increased opacity (white cloudy regions) where the lungs are inflamed or filled with fluid. During training, the network learns to recognize these visual patterns. Each X-ray is resized to 224x224 pixels and passed through the network, which extracts features layer by layer — from simple edges in early layers to complex lung structures in deeper layers. Training was done with PyTorch on the Jetson Orin Nano using `train.py`. The best checkpoint is saved as `model_best.pth`, which stores both the trained weights and the class names. For inference, `test.py` loads this checkpoint, preprocesses the input X-ray, runs it through the model, applies softmax to convert the outputs into probabilities, and prints the predicted class with its confidence percentage.

Dependencies:
PyTorch and torchvision (model training and inference)
Pillow (image loading)
NVIDIA Jetson Orin Nano with JetPack (CUDA acceleration)

 # Running this project
Clone this repository: `git clone https://github.com/Risheek1489-cmd/Pneumonia-project.git`
Install the required libraries: `pip3 install torch torchvision pillow`
Make sure `model_best.pth` is in the project folder (included in this repo).
Set `IMAGE_PATH` in `test.py` to the X-ray image you want to classify (a sample is included).
Run inference: `python3 test.py`
The output prints the prediction (NORMAL or PNEUMONIA) and the confidence percentage.

[View a video explanation here](video link)
