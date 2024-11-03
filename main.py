import os
import io
import logging
import tensorflow.python as tf
import scipy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image



app = Flask(__name__)

# Использование только CPU (если необходимо)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Параметры
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 1

logging.basicConfig(filename='corrupted_images.log', level=logging.ERROR)


def preprocess_image(image_path):

    img = Image.open(image_path)
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Функция для загрузки изображения с обработкой ошибок
def load_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            img = Image.open(io.BytesIO(f.read()))
            img.verify()  # Проверка, что это действительно изображение
        return img
    except Exception as e:
        print(f"Ошибка загрузки изображения {image_path}: {e}")
        return None

# Загрузка данных
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

try:
    train_generator = train_datagen.flow_from_directory(
        '/home/pc/Documents/folder2',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
except Exception as e:
    print(f"Ошибка при создании генератора для тренировочных данных: {e}")

try:
    validation_generator = train_datagen.flow_from_directory(
        '/home/pc/Documents/folder2',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
except Exception as e:
    print(f"Ошибка при создании генератора для валидационных данных: {e}")

from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('my_model.h5')



def predict_animal(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_indices = train_generator.class_indices
    class_labels = list(class_indices.keys())

    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class


def preprocess_image(image_path):

    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'Нет файла изображения'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'Нет выбранного файла'}), 400


        file_path = (r'C:\Users\hackaton\Downloads' + file.filename)
        file.save(file_path)
        img_array = preprocess_image(r'C:\Users\hackaton\Downloads' + file.filename)
        prediction = predict_animal(r'C:\Users\hackaton\Downloads' + file.filename)



        return jsonify({'message': f'Изображение "{prediction}" успешно загружено и обработано.'})
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(r'C:\Users\hackaton\Downloads' + filename)
            img_array = preprocess_image(r'C:\Users\hackaton\Downloads' + filename)
            prediction = predict_animal(r'C:\Users\hackaton\Downloads' + file.filename)

            return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)