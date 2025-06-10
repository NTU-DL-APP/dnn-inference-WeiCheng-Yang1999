import tensorflow as tf
import numpy as np
import json
import os

# 1. 載入 Fashion MNIST 資料集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 2. 正規化並扁平化
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# 3. 建構模型（Dense + ReLU + Softmax）
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. 編譯與訓練
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.1)

# 5. 儲存為 .h5
os.makedirs("model", exist_ok=True)
model.save("model/fashion_mnist.h5")

# 6. 匯出 model architecture 為 JSON
model_json = []
for layer in model.layers:
    cfg = {
        "name": layer.name,
        "type": layer.__class__.__name__.replace("Layer", ""),
        "config": {},
        "weights": []
    }

    if isinstance(layer, tf.keras.layers.Dense):
        cfg["config"]["activation"] = layer.activation.__name__
        weights = layer.get_weights()
        w_name = [layer.name + "_W", layer.name + "_b"]
        cfg["weights"] = w_name
    elif isinstance(layer, tf.keras.layers.Flatten):
        cfg["type"] = "Flatten"

    model_json.append(cfg)

with open("model/fashion_mnist.json", "w") as f:
    json.dump(model_json, f, indent=2)

# 7. 匯出 weights 為 npz
weight_dict = {}
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()
        weight_dict[layer.name + "_W"] = weights[0]
        weight_dict[layer.name + "_b"] = weights[1]

np.savez("model/fashion_mnist.npz", **weight_dict)

print("✅ 優化後模型與參數已儲存至 /model")
