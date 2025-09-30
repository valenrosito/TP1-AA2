import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from recordDataset import extract_hand_xy, normalize_landmarks, LABEL_NAMES

# ---------- Config ----------
MODEL_PATH = "TP1 - Problema 2/gesture_classifier_model.h5"
USE_NORMALIZATION = True

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    # Cargar modelo entrenado
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Modelo cargado desde {MODEL_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            pred_label, probs = None, None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                vec = extract_hand_xy(hand_landmarks)
                if USE_NORMALIZATION:
                    vec = normalize_landmarks(vec)

                if vec.shape == (42,):
                    X = np.expand_dims(vec, axis=0)
                    probs = model.predict(X, verbose=0)[0]
                    pred_label = int(np.argmax(probs))

            # HUD de predicciones
            if pred_label is not None:
                h, w = frame.shape[:2]
                y0 = 30
                cv2.putText(frame, f"Gesto: {LABEL_NAMES[pred_label]}",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2, cv2.LINE_AA)

                # Probabilidades por clase
                for i, name in LABEL_NAMES.items():
                    y0 += 30
                    prob = probs[i] if probs is not None else 0
                    cv2.putText(frame, f"{name}: {prob:.2f}",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

            cv2.imshow("Rock-Paper-Scissors", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
