import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = 'C:/Users/redmi/best_model.keras'
class_names_ru = {
    0: 'Горох', 1: 'Брокколи', 2: 'Капуста', 3: 'Перец', 4: 'Морковь',
    5: 'Цветная капуста', 6: 'Огурец', 7: 'Картофель', 8: 'Тыква', 9: 'Редис', 10: 'Помидор'
}
model = None


def load_model_once():
    global model
    if model is None:
        model = load_model(MODEL_PATH)


def predict_vegetable(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    load_model_once()
    preds = model.predict(x)[0]
    predicted_classes_with_prob = {class_names_ru[i]: float(preds[i]) for i in range(len(preds))}
    return predicted_classes_with_prob


def index(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        img_path = os.path.join(fs.location, filename)

        predictions = predict_vegetable(img_path)

        sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        top_3_predictions = sorted_predictions[:3]

        return render(request, 'classifier/index.html', {
            'uploaded_file_url': uploaded_file_url,
            'predictions': top_3_predictions
        })
    return render(request, 'classifier/index.html')