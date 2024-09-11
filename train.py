import numpy as np
import os
import cv2
from PIL import Image


def train_data(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        try:
            img = Image.open(image).convert('L')
            imagenp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split('.')[1])
            faces.append(imagenp)
            ids.append(id)
        except Exception as e:
            print(f"Error processing file {image}: {e}")

    ids = np.array(ids, dtype=np.int32)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")


train_data('data')