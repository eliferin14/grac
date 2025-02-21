#https://www.samproell.io/posts/ai/asl-detector-with-mediapipe-wsl/#detailed-performance-evaluation

import mediapipe as mp
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer
import argparse
import tqdm
import pandas as pd
import os
import numpy as np

# Configuration
model_path = "exported_model/custom_gesture_recognizer.task"  # Replace with the actual path to your .task file
image_path = "reduced_dataset/none/0a3aa1a5-b1c8-4936-ab74-999f1545e742.jpg"  # Replace with the path to your test image
dataset_path = "conf_mat_dataset"

# Create a GestureRecognizer object
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

recognizer = GestureRecognizer.create_from_options(options)

mp_image = mp.Image.create_from_file(image_path)
result = recognizer.recognize(mp_image)
print(result.gestures[0][0])




classification_results = []

for folder in os.listdir(dataset_path):
    print(f"Analysing folder {folder}")    
    folder_path = os.path.join(dataset_path, folder) 
    
    for filename in os.listdir( folder_path ):
        
        image_path =  os.path.join(folder_path, filename) 
        mp_image = mp.Image.create_from_file(image_path)
        
        result = recognizer.recognize(mp_image)
        
        if len(result.gestures) > 0:
            prediction = result.gestures[0][0].category_name or "n/a"
        else:
            prediction = "n/a"
            
        if prediction == "n/a": continue
        
        print(f"{filename}: {folder} -> {prediction}")
        
        classification_results.append([filename, folder, prediction])
        
        
        
results_df = pd.DataFrame(classification_results, columns=["filename", "label", "pred"])


import sklearn.metrics
import matplotlib.pyplot as plt

classes = results_df['label'].unique().tolist()
cm = sklearn.metrics.confusion_matrix(
    results_df["label"], results_df["pred"], labels=classes, normalize="true"
)
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)  # Customize as needed
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha='right')


plt.show()




import seaborn as sns

results_df["result"] = np.where(
    results_df.pred == results_df.label,
    "correct",
    np.where(results_df.pred.isin(["n/a", "empty"]), "not found", "incorrect"),
)
print(results_df.result.value_counts(normalize=True))
sns.histplot(
    data=results_df, x="label", hue="result", multiple="stack", stat="count"
)
results_df.query("result == 'incorrect'").groupby(
    "label"
).pred.value_counts().sort_values(ascending=False)

# label  pred
# M      N       12
# T      S        4
# N      S        3
# E      S        2
# R      U        2
# C      O        2







plt.show()

