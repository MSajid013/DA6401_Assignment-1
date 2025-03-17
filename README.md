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

# Question-2:
Implements a Feedforward neural network from scratch. first, it defines the activation functions:

