import os
import random
import shutil
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import argparse

import cv2
import numpy as np

def rotate_image(img, angle):
    """Rotates an image by a given angle.

    Args:
        img: The input image as a NumPy array.
        angle: The rotation angle in degrees.

    Returns:
        The rotated image.
    """

    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image   

parser = argparse.ArgumentParser(description="Hello")
parser.add_argument("-dp", "--dataset_path", type=str, default='dataset', help="Dataset path")
parser.add_argument("--use_pretrained", action='store_true', help="If True the pretrained model is further trained")

args = parser.parse_args()
print(args)

# Delete the old dataset and model
try:        
    if not args.use_pretrained:
        shutil.rmtree("exported_model")
except OSError as e:
    print(f"Error: {e.strerror}")

dp = args.dataset_path

labels = []
for i in os.listdir(dp):
  if os.path.isdir(os.path.join(dp, i)):
    labels.append(i)
print(labels)

data = gesture_recognizer.Dataset.from_folder(
    dirname=dp,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

hparams = gesture_recognizer.HParams(export_dir="exported_model", epochs=30)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

model.export_model('custom_gesture_recognizer.task')
model.export_labels('exported_model')