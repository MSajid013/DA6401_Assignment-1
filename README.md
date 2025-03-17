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

Then loads Fashion MNIST data and preprocesses it by normalizing and flattening images, and converts labels into one-hot encoded format. After that initializes the neural network with 5 hidden layers, 128 neurons per layer and 'xavier' weight initialization, then runs forward propagation with 'sigmoid' activation on a single sample and outputs the predicted class probabilities.

# Question-3: Overview
Implements cross-entropy loss computation, backward propagation and accuracy evaluation for training a multi-layer neural network. 
1. Compute Cross-Entropy Loss (cross_entropy_loss): This function calculates the cross-entropy loss for multi-class classification with L2 regularization (weight decay) to prevent overfitting.

2. Backward Propagation (backward_propagation): The backpropagation algorithm computes gradients for weights and biases by propagating errors backward through the network and support for the following optimisation functions:
* Stochastic Gradient Descent (stochastic_gd)
* Momentum Based Gradient Descent (momentum_optimizer)
* Nesterov Accelerated Gradient Descent (nag_optimizer)
* RMSprop (rmsprop_optimizer)
* ADAM (adam_optimizer)
* NADAM (nadam_optimizer)

3. Compute Accuracy (compute_accuracy): Evaluates the classification accuracy of the model (ratio of correct predictions to total samples).

# Question-4: Overview
The model is trained with various optimizers, activation functions and hyperparameter settings through a Bayesian sweep. It trains a neural network on the Fashion MNIST dataset and optimizes its hyperparameters using Weights & Biases (WandB). 

Used the standard train/test split of fashion_mnist ( (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() ). Kept 10% of the training data aside as validation data for this hyperparameter search. The images are flattened into 1D arrays of size 784, pixel values are normalized to the range [0,1] and labels are one-hot encoded for multi-class classification.

Model Training Function: train_model() : This function trains a neural network with customizable hyperparameters:

Sweep Configuration: Used Bayesian optimization method to efficiently search the hyperparameter space. The parameters tested include:
* Epochs: [5, 10]
* Number of layers: [3, 4, 5]
* Neurons per layer: [32, 64, 128]
* weight decay (L2 regularisation): [0, 0.0005, 0.5]
* Learning rate: [0.001, 0.0001]
* Batch size: [16, 32, 64]
* Optimizers: ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
* Activation functions: ['sigmoid', 'relu', 'tanh']
* Weight initialization: ['random', 'xavier']
  
Then initialized the sweep using wandb.sweep(sweep_config, project='DA6401_Assignment-1') with sweep name (sweep cross entropy now), the main() function retrieves hyperparameter values assigned by WandB and trains the model and the wandb.agent(sweep_id, function=main, count=100) command runs 100 training jobs with different hyperparameter combinations.

After that, getting higher accuracy with 10 epochs, [4,5] number of layers, [64, 128] neurons per layer, [32, 64] batch size, ['rmsprop', 'adam', 'nadam'] optimizers, ['sigmoid', 'relu'] activation functions and 'xavier' weight initialization. So again initialized the sweep using wandb.sweep(sweep_config, project='DA6401_Assignment-1') with sweep name (sweep cross entropy later) with above configuration and the wandb.agent(sweep_id, function=main, count=50) command runs 50 training jobs with different hyperparameter combinations.

# Question-5: Overview
Got 87.433% best accuracy on the validation set with hyperparameters (10 epochs, 5 number of layers, 64 neurons per layer, 0.0005 regularisation, 0.0001 learning rate, 64 batch size, 'rmsprop' optimizers, 'relu' activation functions and 'xavier' weight initialization) which is shown in the attached plot in WandB report.

# Question-6: Overview
Attached "Parallel co-ordinates plot" and "correlation summary" and wrote my observations in the WandB report.

# Question-7: Overview
Used NADAM, ADAM and RMSprop optimizers for computing test accuracy with best hyperparameters and got 86.51% highest test accuracy with RMSprop optimizer, 'relu' activation and 'xavier' weight initialization.

Plotted the confusion matrix with RMSprop optimiser, 'relu' activation and 'xavier' weight initialization. Mentioned these hyperparameters in WandB report.

# Question-8: Overview
Compared the cross entropy loss with the squared error loss by plotting the loss over epochs. Used NADAM, ADAM and RMSprop optimizers for computing training accuracy and test accuracy with best hyperparameters.
Same configuration used in question-7 as well and can be compared that, getting slightly higher accuracy with cross entropy loss. I Mentioned these configurations in the WandB report.

# Question-9: Overview
Attached my github repo link in the WandB report.

# Question-10: Overview
After tuning with WandB, Used 3 best hyperparameter configurations for MNIST dataset. Got 96.73% highest test accuracy on MNIST dataset. Mentioned these hyperparameter configurations and reported all accuracies with these configurations in the WandB report.






