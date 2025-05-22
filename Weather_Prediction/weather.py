import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, ConvLSTM2D, BatchNormalization, Conv3D,
                                     Layer, Multiply, Permute, Reshape, Softmax)
import matplotlib.pyplot as plt
import os

# ====== 1. Load and Preprocess Data ======
def load_data(file_path, var_name='t2m', time_steps=10):
    ds = xr.open_dataset(file_path)
    data = ds[var_name].values  # shape: (time, lat, lon)
    data = (data - np.mean(data)) / np.std(data)  # normalize
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
    return np.expand_dims(np.array(sequences), axis=-1)  # shape: (samples, time, lat, lon, 1)

# ====== 2. Optional: Attention Layer ======
class SpatialAttention(Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()

    def call(self, inputs):
        # inputs: (batch, time, H, W, C)
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        score = tf.keras.layers.Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(concat)
        return Multiply()([inputs, score])

# ====== 3. Model Definition ======
def build_model(input_shape, use_attention=False):
    inputs = Input(shape=input_shape)  # (T, H, W, C)

    x = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)

    if use_attention:
        x = SpatialAttention()(x)

    x = ConvLSTM2D(filters=16, kernel_size=(3, 3),
                   padding='same', return_sequences=False)(x)
    x = BatchNormalization()(x)

    output = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                    activation='linear', padding='same')(x)

    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ====== 4. Training ======
def train_model(model, X, y, epochs=10, batch_size=4):
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return history

# ====== 5. Evaluation ======
def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    error = np.mean((pred - y_test) ** 2)
    print("MSE on test set:", error)

    plt.subplot(1, 2, 1)
    plt.imshow(y_test[0, :, :, 0], cmap='coolwarm')
    plt.title("True")

    plt.subplot(1, 2, 2)
    plt.imshow(pred[0, :, :, 0], cmap='coolwarm')
    plt.title("Predicted")
    plt.show()

# ====== 6. Main Script ======
if __name__ == '__main__':
    # Replace with your actual NetCDF file path and variable
    data_path = 'your_era5_or_noaa_file.nc'
    var_name = 't2m'  # e.g., '2m_temperature' for ERA5, 'prcp' for NOAA
    time_steps = 10

    print("Loading data...")
    X = load_data(data_path, var_name, time_steps)
    y = X[:, -1, :, :, :]  # last frame as target
    X = X[:, :-1, :, :, :]  # input: previous frames

    print("Building model...")
    model = build_model(input_shape=X.shape[1:], use_attention=True)

    print("Training model...")
    train_model(model, X, y, epochs=5, batch_size=2)

    print("Evaluating model...")
    evaluate_model(model, X[-5:], y[-5:])
