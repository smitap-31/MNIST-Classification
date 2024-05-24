import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import mlflow
import mlflow.tensorflow

# Load and preprocess the MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    
    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow.tensorflow.autolog()  # Automatically log parameters, metrics, and model
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        mlflow.log_artifact('mnist_model.h5')

    model.save('mnist_model.h5')
    print(f"Model saved as mnist_model.h5 and logged to MLflow with run ID {run.info.run_id}")

if __name__ == "__main__":
    train()
