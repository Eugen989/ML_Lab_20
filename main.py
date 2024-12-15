import pickle

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"}]

loaded_model_knn = pickle.load(open('model/Iris_pickle_file', 'rb'))
# Загрузка весов из файла
new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('model/neuron_weights.txt')
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
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Помидор', 'Огурец'))
        return render_template('lab4.html', title="Первый нейрон", menu=menu,
                               class_model="Это: " + str(*np.where(predictions >= 0.5, 'Помидор', 'Огурец')))

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
    predictions = model_reg.predict(input_data)

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
