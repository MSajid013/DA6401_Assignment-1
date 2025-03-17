## DA6401_Assignment-1
# Question-1: Overview

The code initializes a WandB project (DA6401_Assignment-1) to track and visualize data. It loads the Fashion-MNIST dataset, which has images of ten different clothing items and picks one sample image for each class. A function (plot_sample) is defined to randomly select and show seven images with their labels. Then these images are uploaded to WandB for visualization. At the end, the Wandb session is closed.

Dataset: The Fashion MNIST dataset consists of grayscale images of ten different classes of clothing items:
1. Top/Tshirt
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankleboot

The dataset is loaded from tensorflow.keras.datasets .

# Question-2: Overview
Implements a Feedforward neural network from scratch using numpy which takes images from Fashion MNIST dataset as input and outputs a probability distribution over the 10 classes. The network supports different activation functions, weight initialization methods and multiple hidden layers. It is enough flexible such that it is easy to change the number of hidden layers and the number of neurons in each hidden layer.

Features:
1. Implements three activation functions (Sigmoid, Relu, Tanh) and their derivatives which required for backpropagation.
2. Defines softmax function  which converts logits into probabilities for multiclass classification.
3. Used two weight initialization ('random' or 'xavier') which initializes weights and biases using either 'random' or 'xavier' initialization..
4. Implements forward propagation through hidden layers to compute activations and final output layer using softmax to compute predictions.

Then loads Fashion MNIST data and preprocesses it by normalizing and flattening images, and converts labels into one-hot encoded format. After that initializes the neural network with 5 hidden layers and 128 neurons per layer, then runs forward propagation on a single sample and outputs the predicted class probabilities.

