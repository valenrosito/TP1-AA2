# record-dataset.py
import os
import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict

# ---------- Config ----------
SAVE_DATA_PATH = "TP1 - Problema 2/datasets/rps_dataset.npy"
SAVE_LABELS_PATH = "TP1 - Problema 2/datasets/rps_labels.npy"
# Normalización: centra en la muñeca (landmark 0) y escala por el mayor rango (x o y)
USE_NORMALIZATION = True

# Mapas de etiquetas
LABELS = {ord('0'): 0, ord('1'): 1, ord('2'): 2}
LABEL_NAMES = {0: "piedra", 1: "papel", 2: "tijeras"}

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_xy(hand_landmarks):
    """
    Extrae un vector de longitud 42: [x0,y0, x1,y1, ..., x20,y20]
    Coordenadas normalizadas a [0,1] respecto del frame.
    """
    coords = []
    for lm in hand_landmarks.landmark:
        x = lm.x  # proporción [0,1] en ancho
        y = lm.y  # proporción [0,1] en alto
        coords.extend([x, y])
    return np.array(coords, dtype=np.float32)

def normalize_landmarks(vec_xy):
    """
    Normaliza el vector 42 (x,y)*21:
    - Traslada para que la muñeca (índice 0) quede en (0,0)
    - Escala por el mayor rango (max abs en x/y) para mantener proporciones
    """
    vec = vec_xy.reshape(-1, 2)  # (21, 2)
    wrist = vec[0].copy()
    vec -= wrist  # trasladar
    scale = np.max(np.abs(vec))  # mayor magnitud en cualquier eje
    if scale < 1e-6:
        scale = 1.0
    vec /= scale
    return vec.flatten()

def draw_hud(frame, counts_by_label, tip="0: piedra | 1: papel | 2: tijeras | n: toggle norm | s: guardar | ESC: salir", norm_on=True):
    h, w = frame.shape[:2]
    # Caja HUD
    cv2.rectangle(frame, (10, 10), (w-10, 110), (0, 0, 0), -1)
    cv2.putText(frame, tip, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Normalizacion: {'ON' if norm_on else 'OFF'}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
    info = " | ".join([f"{LABEL_NAMES[i]}: {counts_by_label[i]}" for i in [0,1,2]])
    cv2.putText(frame, info, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

def load_existing_dataset():
    """Carga dataset previo si existe, sino devuelve vacíos con shapes correctas."""
    if os.path.exists(SAVE_DATA_PATH) and os.path.exists(SAVE_LABELS_PATH):
        try:
            old_X = np.load(SAVE_DATA_PATH)
            old_y = np.load(SAVE_LABELS_PATH)
            # Validaciones básicas
            if old_X.ndim != 2 or old_X.shape[1] != 42:
                print("[WARN] rps_dataset.npy con forma inesperada; se ignorará el dataset previo.")
                return np.empty((0, 42), dtype=np.float32), np.empty((0,), dtype=np.int64)
            if old_y.ndim != 1 or old_X.shape[0] != old_y.shape[0]:
                print("[WARN] Inconsistencia entre datos y labels previos; se ignorarán.")
                return np.empty((0, 42), dtype=np.float32), np.empty((0,), dtype=np.int64)
            print(f"[INFO] Cargando dataset existente: X{old_X.shape}, y{old_y.shape}")
            return old_X.astype(np.float32, copy=False), old_y.astype(np.int64, copy=False)
        except Exception as e:
            print(f"[WARN] No se pudo cargar dataset previo ({e}); se empezará desde cero.")
    return np.empty((0, 42), dtype=np.float32), np.empty((0,), dtype=np.int64)

def save_concat_dataset(new_data, new_labels):
    """Concatena con el dataset previo si existe y guarda en los mismos archivos."""
    old_X, old_y = load_existing_dataset()
    # Acomodar tipos
    new_X = np.array(new_data, dtype=np.float32).reshape(-1, 42) if len(new_data) else np.empty((0, 42), dtype=np.float32)
    new_y = np.array(new_labels, dtype=np.int64).reshape(-1,) if len(new_labels) else np.empty((0,), dtype=np.int64)

    X = np.concatenate([old_X, new_X], axis=0)
    y = np.concatenate([old_y, new_y], axis=0)

    np.save(SAVE_DATA_PATH, X)
    np.save(SAVE_LABELS_PATH, y)
    print(f"[SAVE] Guardado -> {SAVE_DATA_PATH}: {X.shape}, {SAVE_LABELS_PATH}: {y.shape}")
    print("       Por clase (acumulado):", {LABEL_NAMES[i]: int((y == i).sum()) for i in [0,1,2]})

def main():
    data = []   # lista de vectores (42,)
    labels = [] # lista de ints {0,1,2}
    counts = defaultdict(int)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Para dataset, conviene 1 mano por frame
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        global USE_NORMALIZATION
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            # Dibujar landmarks si hay mano
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

            # HUD
            draw_hud(frame, counts, norm_on=USE_NORMALIZATION)

            cv2.imshow('Grabación dataset RPS - ESC para salir', frame)
            key = cv2.waitKey(1) & 0xFF

            # Salir
            if key == 27:  # ESC
                break

            # Toggle normalización
            if key in (ord('n'), ord('N')):
                USE_NORMALIZATION = not USE_NORMALIZATION

            # Guardar muestra del frame actual con etiqueta
            if key in LABELS:
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    vec = extract_hand_xy(hand_landmarks)
                    if USE_NORMALIZATION:
                        vec = normalize_landmarks(vec)
                    if vec.shape == (42,):
                        y = LABELS[key]
                        data.append(vec)
                        labels.append(y)
                        counts[y] += 1
                        print(f"[OK] Capturado gesto '{LABEL_NAMES[y]}' | total sesión: {counts[y]}")
                    else:
                        print("[WARN] Vector de landmarks con tamaño inesperado:", vec.shape)
                else:
                    print("[INFO] No se detectó mano; no se guardó muestra.")

            # Guardado parcial (concatena con lo existente)
            if key in (ord('s'), ord('S')):
                if len(data) > 0:
                    save_concat_dataset(data, labels)
                else:
                    print("[INFO] No hay datos nuevos en esta sesión para guardar aún.")

    # --------- Cierre: guardar definitivo (concatena) ---------
    cap.release()
    cv2.destroyAllWindows()

    if len(data) == 0:
        print("[FIN] No se capturaron muestras nuevas. No se modificó el dataset.")
        return

    save_concat_dataset(data, labels)
    print("[FIN] Sesión guardada y concatenada correctamente.")

if __name__ == "__main__":
    main()
