# train_model.py

import tensorflow as tf
import numpy as np
import os
import json

# === 訓練參數設定 ===
EPOCHS = 10
BATCH_SIZE = 128
MODEL_DIR = "model"
ARCH_PATH = os.path.join(MODEL_DIR, "fashion_mnist.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "fashion_mnist.npz")

# === 1. 載入資料集 ===
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# === 2. 建立模型（符合作業限制）===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu', name="dense_1"),
    tf.keras.layers.Dense(64, activation='relu', name="dense_2"),
    tf.keras.layers.Dense(10, activation='softmax', name="dense_3")
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 3. 訓練模型 ===
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# === 4. 測試準確率 ===
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# === 5. 匯出模型架構 ===
os.makedirs(MODEL_DIR, exist_ok=True)
with open(ARCH_PATH, "w") as f:
    f.write(model.to_json())
print(f"✅ Saved model architecture to {ARCH_PATH}")

# === 6. 匯出模型權重為 npz（照作業格式）===
weights = {}
for layer in model.layers:
    if "dense" in layer.name:
        W, b = layer.get_weights()
        weights[f"{layer.name}_W"] = W
        weights[f"{layer.name}_b"] = b

np.savez(WEIGHTS_PATH, **weights)
print(f"✅ Saved model weights to {WEIGHTS_PATH}")
