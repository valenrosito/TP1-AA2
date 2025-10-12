import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
import matplotlib.pyplot as plt


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

SAVE_DATA_PATH = "TP1 - Problema 2/datasets/rps_dataset.npy"
SAVE_LABELS_PATH = "TP1 - Problema 2/datasets/rps_labels.npy"
MODEL_PATH = "TP1 - Problema 2/gesture_classifier_model.h5"

def load_existing_dataset():
    if os.path.exists(SAVE_DATA_PATH) and os.path.exists(SAVE_LABELS_PATH):
        try:
            data = np.load(SAVE_DATA_PATH)
            labels = np.load(SAVE_LABELS_PATH)
            print(f"Cargado dataset existente: {data.shape[0]} muestras.")
            return data, labels
        except Exception as e:
            print(f"Error cargando dataset existente: {e}")
    return np.empty((0, 42), dtype=np.float32), np.empty((0,), dtype=np.int32)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(42,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(data, labels):
    if data.size == 0:
        print("No hay datos para entrenar.")
        return None

    # Dtypes seguros
    data = data.astype(np.float32, copy=False)
    labels = labels.astype(np.int32, copy=False)

    # Barajar por si grabaste en bloques por clase
    idx = np.random.permutation(len(data))
    data, labels = data[idx], labels[idx]

    model = build_model()

    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    ]

    model.fit(
        data, labels,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=cbs,
        verbose=1
    )

    history = model.history.history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history['val_loss'], label='Pérdida de validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history['val_accuracy'], label='Precisión de validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Guardar (por si el checkpoint no se activó)
    model.save(MODEL_PATH)
    print(f"Modelo entrenado y guardado en '{MODEL_PATH}'")
    return model

if __name__ == "__main__":
    data, labels = load_existing_dataset()
    train_model(data, labels)
