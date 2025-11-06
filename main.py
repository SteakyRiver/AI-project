import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import PIL

# =============== 1. Скачивание и РУЧНАЯ распаковка ===============
print("📥 Скачиваем датасет 'Cats vs Dogs' от Microsoft...")
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
zip_path = tf.keras.utils.get_file("cats_and_dogs.zip", origin=url, extract=False)  # extract=False!

# Определяем путь к папке датасета
base_dir = os.path.dirname(zip_path)
extract_dir = os.path.join(base_dir, "PetImages")

# Распаковываем вручную
if not os.path.exists(extract_dir):
    print("📦 Распаковываем архив...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)
    print(f"✅ Архив распакован в: {extract_dir}")
else:
    print(f"📁 Папка уже существует: {extract_dir}")

# Проверим, что папки Cat и Dog существуют
cat_dir = os.path.join(extract_dir, "Cat")
dog_dir = os.path.join(extract_dir, "Dog")

if not (os.path.exists(cat_dir) and os.path.exists(dog_dir)):
    raise RuntimeError(f"❌ Ожидались папки 'Cat' и 'Dog' в {extract_dir}, но их нет!")

# =============== 2. Очистка от битых изображений ===============
def remove_corrupted_images(folder):
    print(f"🧹 Проверяем папку: {folder}")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            with PIL.Image.open(file_path) as img:
                img.verify()  # Проверка целостности
        except (IOError, SyntaxError, PIL.UnidentifiedImageError, OSError) as e:
            print(f"❌ Удаляем битый файл: {file_path}")
            try:
                os.remove(file_path)
            except Exception as del_err:
                print(f"⚠️ Не удалось удалить {file_path}: {del_err}")

remove_corrupted_images(cat_dir)
remove_corrupted_images(dog_dir)

# =============== 3. Подготовка генераторов ===============
print("⚙️ Настраиваем генераторы данных...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

IMG_SIZE = 128
BATCH_SIZE = 32

train_gen = datagen.flow_from_directory(
    extract_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    extract_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# =============== 4. Модель ===============
print("🧠 Создаём модель...")
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =============== 5. Обучение ===============
print("🚀 Обучение...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=callbacks)

# =============== 6. Сохранение ===============
val_loss, val_acc = model.evaluate(val_gen)
print(f"✅ Точность: {val_acc:.4f}")
model.save("cats_vs_dogs_model.h5")
print("💾 Модель сохранена!")
