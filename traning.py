# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# ========================
# PARAMETER
# ========================
IMG_SIZE = 224
BATCH_SIZE = 16

train_dir = "dataset2/Train"
test_dir = "dataset2/Test"

# ========================
# PREPROCESSING DATA
# ========================
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Dataset berhasil dimuat")
print("Label kelas:", train_data.class_indices)

# ========================
# MODEL MobileNetV2
# ========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze layer
for layer in base_model.layers:
    layer.trainable = False

# Tambah layer sendiri
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ========================
# COMPILE MODEL
# ========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# TRAINING
# ========================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# ========================
# SIMPAN MODEL
# ========================
model.save("model_kemasan.h5")

print("Training selesai & model disimpan ✅")
