import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

model_dir = './dataset/alzheimer_cnn_model'
model = tf.keras.models.load_model(model_dir)

CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
IMG_SIZE = 176

def analyze_image(image):
    # Preprocess the image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform predictions using the loaded model
    predictions = model.predict(image)

    # Process the prediction results and return the analysis
    pred_class = np.argmax(predictions)
    pred_label = CLASSES[pred_class]

    return pred_label

@app.route('/', methods=['GET', 'POST'])

def home():
    # return render_template("index.html")
    if request.method == 'POST':
        image = request.files['image']
        image = Image.open(image)
        result = analyze_image(image)
        return render_template('result.html', result=result)
    else:
        request.method == 'GET'
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
