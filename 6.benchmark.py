import os
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from pathlib import Path

# ---------------- Configuration ----------------
MODEL_PATHS = [
    "saved_model/efficientnet_best_float32.tflite",
    "saved_model/efficientnet_best_float16.tflite",
    "saved_model/efficientnet_best_int8_full.tflite"
]

VAL_DIR = "/home/leanse/AI/interview/clearquote/dataset/test"
IMG_SIZE = 224
NUM_CLASSES = 7  # adjust to your dataset
NUM_THREADS = 4  # CPU threads
# ------------------------------------------------

def preprocess(img_path, input_type, input_scale=1.0, input_zero_point=0):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    if input_type == np.float32:
        return np.expand_dims(img, axis=0)
    elif input_type == np.int8:
        # Proper quantization using scale & zero_point
        img_int8 = np.round(img / input_scale + input_zero_point).astype(np.int8)
        return np.expand_dims(img_int8, axis=0)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

def evaluate_model(model_path):
    # Initialize interpreter
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()

    # Model info
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("\n=== Model:", model_path, "===")
    print("Input details:", input_details)
    print("Output details:", output_details)

    input_index = input_details['index']
    output_index = output_details['index']
    input_type = input_details['dtype']
    input_scale, input_zero_point = input_details.get('quantization', (1.0, 0))

    total = 0
    correct = 0
    times = []

    for class_idx, class_dir in enumerate(sorted(os.listdir(VAL_DIR))):
        class_path = Path(VAL_DIR) / class_dir
        for img_file in class_path.iterdir():
            img = preprocess(str(img_file), input_type, input_scale, input_zero_point)

            start = time.time()
            interpreter.set_tensor(input_index, img)
            interpreter.invoke()
            end = time.time()
            times.append(end - start)

            output = interpreter.get_tensor(output_index)
            pred = np.argmax(output)
            if pred == class_idx:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    avg_latency = (sum(times) / len(times)) * 1000  # ms per image

    print(f"{model_path} Accuracy: {accuracy:.2f}%, Avg inference time: {avg_latency:.2f} ms/image")
    return accuracy, avg_latency

if __name__ == "__main__":
    results = []
    for m in MODEL_PATHS:
        acc, latency = evaluate_model(m)
        results.append((m, acc, latency))

    print("\n=== Benchmark Summary ===")
    for r in results:
        print(f"Model: {r[0]}")
        print(f"  Accuracy: {r[1]:.2f}%")
        print(f"  Avg latency: {r[2]:.2f} ms per image")
