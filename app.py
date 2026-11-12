
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('frank_model.keras')
labels = ['Dog', 'Cat']

def prepare(image_file):
    img = Image.open(image_file).convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            pred = model.predict(prepare(f))
            idx = np.argmax(pred[0])
            result = f"This is a: {labels[idx]} (%{np.max(pred[0])*100:.2f})"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(port=5000)
