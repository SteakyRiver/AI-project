import os
from PIL import Image
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_PATH = 'dataset/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else 'categorical'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f'Ошибка: файл не найден по пути {image_path}')
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f'Ошибка: повреждено изображение - {image_path}')
        return
    model = tf.keras.models.load_model('cats_vs_dogs_model.h5')
    img = cv2.imread(image_path)

    if img is None:
        print(f'Ошибка: не удалось прочитать изображение - {image_path}')
        return
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_names = os.listdir(DATASET_PATH)
    if class_mode == 'binary':
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.artgmax(prediction, axis = -1).numpy()[0]]
    print(f'Модель определила: {predicted_class}')

    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f'класс: {predicted_class}')
    plt.axis('off')
    plt.show()

predict_image('test_dog.jpg')