import pickle
import os

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"}]

loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))
# Загрузка весов из файла
new_neuron = tf.keras.models.load_model('model/dogs.keras')
model_reg = tf.keras.models.load_model('model/regression_model.h5')
model_class = tf.keras.models.load_model('model/classification_model.h5')

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)

@app.route("/p_lab2")
def f_lab2():
    return render_template('lab2.html', title="Логистическая регрессия", menu=menu)


@app.route("/p_lab3")
def f_lab3():
    return render_template('lab3.html', title="Логистическая регрессия", menu=menu)

@app.route("/p_lab4", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')

    if request.method == 'POST':
        if 'img_cloth' not in request.files:
            return 'Нет файла'

        file = request.files['img_cloth']

        if file.filename == '':
            return 'Нет выбранного файла'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            img = image.load_img(filepath, target_size=(28, 28), color_mode='grayscale')
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = new_neuron.predict_on_batch(img_array)

            predictions = new_neuron.predict_on_batch(img_array).flatten()
            predictions = tf.nn.sigmoid(predictions)
            predictions = tf.where(predictions < 0.5, 0, 1)
            print(score)
            return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model=f"Это изображение похоже на {class_names[int(predictions)]}")

        else:
            return 'Недопустимый тип файла'

@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('sepal_length')),
                       float(request.args.get('sepal_width')),
                       float(request.args.get('petal_length')),
                       float(request.args.get('petal_width'))]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(sort=pred[0])

@app.route('/api_v2', methods=['get'])
def get_sort_v2():
    request_data = request.get_json()
    X_new = np.array([[float(request_data['sepal_length']),
                       float(request_data['sepal_width']),
                       float(request_data['petal_length']),
                       float(request_data['petal_width'])]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(sort=pred[0])


@app.route('/api_reg', methods=['get'])
def predict_regression():
    # Получение данных из запроса http://localhost:5000/api_reg?celsius=100
    input_data = np.array([float(request.args.get('celsius'))])
    # input_data = np.array(input_data.reshape(-1, 1))

    # Предсказание
    predictions = model_reg.predict(input, _data)

    return jsonify(fahrenheit=str(predictions))

@app.route('/api_class', methods=['get'])
def predict_classification():
    # Получение данных из запроса http://localhost:5000/api_class?width=5&length=5
    input_data = np.array([[float(request.args.get('width')),
                       float(request.args.get('length'))]])
    print(input_data)
    # input_data = np.array(input_data.reshape(-1, 1))

    # Предсказание
    predictions = model_class.predict(input_data)
    print(predictions)
    result = 'Помидор' if predictions >= 0.5 else 'Огурец'
    print(result)
    # меняем кодировку
    app.config['JSON_AS_ASCII'] = False
    return jsonify(ov = str(result))

if __name__ == "__main__":
    app.run(debug=True)
