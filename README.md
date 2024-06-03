# Wiser_test
The provided code builds and trains a convolutional neural network (CNN) to detect emotions from facial images. Initially, necessary libraries from TensorFlow and Keras are imported to facilitate model building, training, and data preprocessing. The core of the CNN is defined in the create_model function, which constructs a sequential model comprising multiple convolutional layers with ReLU activation functions for feature extraction. These layers are interspersed with max pooling layers that reduce the spatial dimensions of the data, thereby decreasing the computational load and highlighting the most critical features.

The network architecture includes three convolutional layers with 32, 64, and 128 filters, respectively, each followed by a max pooling layer. The extracted features are then flattened into a one-dimensional vector, which is processed by a fully connected dense layer with 256 units. A dropout layer is included to prevent overfitting by randomly disabling 50% of the neurons during training. The final output layer uses a softmax activation function to classify the input images into one of seven emotion categories.

After defining the model, it is compiled with the Adam optimizer, which is well-suited for handling sparse gradients and noisy data, and the categorical cross-entropy loss function, appropriate for multi-class classification problems. Accuracy is set as the primary metric to evaluate the model’s performance.

For training, ImageDataGenerator is used to augment the training dataset, applying random transformations such as rotation, width and height shifts, shear, zoom, and horizontal flips. This augmentation increases the diversity of the training data, aiding the model in generalizing better to unseen data. The training data is rescaled to normalize pixel values to the range [0, 1]. Both the training and validation data are loaded from directories using the flow_from_directory method, which generates batches of augmented data on the fly.

The model is then trained using the fit method, with early stopping implemented to monitor the validation accuracy. This callback function halts training if the validation accuracy does not improve for five consecutive epochs, restoring the model weights from the epoch with the best validation accuracy, thus preventing overfitting and reducing training time.

Finally, the trained model is evaluated on the validation set to determine its performance, and the validation accuracy is printed. The model is also evaluated on a separate test set to assess its generalization capability on completely unseen data. The test accuracy is printed to give a final measure of the model’s effectiveness in emotion detection.

The code is designed with clear documentation and modularity, making it easy to understand and execute. Instructions are provided for setting up dependencies, organizing the dataset, and running the training and evaluation scripts, ensuring that users can replicate the results without difficulty.

Directions of Use:

Clone the repository.
Place the dataset in the appropriate directories (train, val, test).
Run the training script python3 Wiser_Test.py.
