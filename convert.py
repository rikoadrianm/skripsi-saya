from PIL import Image
import os

folder = "Dataset/NG"

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    try:
        img = Image.open(path)
        img.verify()
    except:
        print("Gambar rusak:", file)
