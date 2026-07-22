import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 16

train_dir = "dataset2/Train"
test_dir = "dataset2/Test"

train_datagen = ImageDataGenerator(
    rescale=1./255)
test_datagen = ImageDataGenerator(
    rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)
print("Dataset berhasil dimuat")
print("Label kelas:", train_data.class_indices)
