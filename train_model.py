import tensorflow as tf
import numpy as np
import json
import os

# === 訓練參數 ===
EPOCHS = 25
BATCH_SIZE = 128
MODEL_DIR = "model"
ARCH_PATH = os.path.join(MODEL_DIR, "fashion_mnist.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "fashion_mnist.npz")

# === 1. 載入 Fashion-MNIST ===
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

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

# === 4. 測試結果（可選）===
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"🎯 Test accuracy: {test_acc:.4f}")

# === 5. 匯出 JSON 架構（符合 nn_predict 格式） ===
os.makedirs(MODEL_DIR, exist_ok=True)
model_json = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        cfg = {
            "name": layer.name,
            "type": "Dense",
            "config": {
                "activation": layer.activation.__name__
            },
            "weights": [f"{layer.name}_W", f"{layer.name}_b"]
        }
        model_json.append(cfg)

with open(ARCH_PATH, "w") as f:
    json.dump(model_json, f, indent=2)
print(f"✅ 模型架構已儲存至 {ARCH_PATH}")

# === 6. 匯出對應權重 npz（符合格式） ===
weight_dict = {}
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        W, b = layer.get_weights()
        weight_dict[f"{layer.name}_W"] = W
        weight_dict[f"{layer.name}_b"] = b

np.savez(WEIGHTS_PATH, **weight_dict)
print(f"✅ 模型權重已儲存至 {WEIGHTS_PATH}")
