import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 16

test_dir = "dataset2/Test"

# Load model
model = tf.keras.models.load_model("model_kemasan.h5")

# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Prediksi
y_pred = model.predict(test_data)

# Ambil kelas dengan probabilitas terbesar
y_pred_classes = np.argmax(y_pred, axis=1)

# Label asli
y_true = test_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=list(test_data.class_indices.keys())
    )
)
