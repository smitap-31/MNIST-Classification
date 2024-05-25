import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import mlflow

app = Flask(__name__)
model = load_model('mnist_model.h5')

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file)
    img = preprocess_image(img)
    
    with mlflow.start_run() as run:
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)
        mlflow.log_param("input_file", file.filename)
        mlflow.log_metric("predicted_class", int(predicted_class[0]))
        return jsonify({'digit': int(predicted_class[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)