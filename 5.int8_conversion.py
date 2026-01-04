import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path

# ---------------- Config ----------------
SAVED_MODEL_DIR = "saved_model"             # Your FP32 saved model
TFLITE_INT8_PATH = "efficientnet_best_int8_full.tflite"
CALIB_DIR = "calibration_folder"            # Representative images for calibration
IMG_SIZE = 224                              # Input size of your model
# ----------------------------------------

# Representative dataset generator
def representative_dataset():
    files = sorted(os.listdir(CALIB_DIR))
    for file in files:
        img_path = os.path.join(CALIB_DIR, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0       # normalize [0,1]
        img = np.expand_dims(img, axis=0)
        yield [img]

# Load FP32 saved model and convert to full INT8
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Set full INT8 quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert and save
tflite_model = converter.convert()
with open(TFLITE_INT8_PATH, "wb") as f:
    f.write(tflite_model)

print(f"INT8 FULL model created successfully: {TFLITE_INT8_PATH}")
