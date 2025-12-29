import cv2
import numpy as np
import glob
import os

from swap_caras_foto import preparar_foto, aplicar_swap as aplicar_swap_foto
from prueba2 import capturar_plantillas_dos_caras, aplicar_swap_dos_caras

PASSWORD = [1, 3, 2]
MODE_LOCKED = 0
MODE_PHOTO = 1
MODE_TWO = 2

SKIN_LOW = np.array([0, 30, 60], dtype=np.uint8)
SKIN_HIGH = np.array([20, 170, 255], dtype=np.uint8)


def load_photo_paths(folder):
    paths = []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    return paths


def detect_gestures(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.2):h, int(w * 0.5):w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_LOW, SKIN_HIGH)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 8000]

    two_hands = len(contours) >= 2
    digit = None

    if len(contours) == 0:
        return None, two_hands

    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return None, two_hands

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return None, two_hands

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d < 3000:
            continue
        start = cnt[s][0].astype(np.float32)
        end = cnt[e][0].astype(np.float32)
        far = cnt[f][0].astype(np.float32)

        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(far - end)
        if b == 0 or c == 0:
            continue
        cos_angle = (b * b + c * c - a * a) / (2 * b * c)
        cos_angle = max(-1.0, min(1.0, float(cos_angle)))
        angle = np.degrees(np.arccos(cos_angle))
        if angle < 90:
            finger_count += 1

    if finger_count == 0 and len(contours) > 0:
        digit = 1
    else:
        digit = min(finger_count + 1, 5)

    return digit, two_hands


def main(camera_index=0, width=1280, height=720, photo_folder="fotos"):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    mode = MODE_LOCKED
    password_input = []
    prev_digit = None
    prev_two_hands = False

    photo_paths = load_photo_paths(photo_folder)
    photo_idx = 0

    current_photo = None
    current_mask = None
    current_src_pts = None
    prev_dst_pts_photo = None

    if photo_paths:
        current_photo, current_mask, current_src_pts = preparar_foto(photo_paths[photo_idx])

    templates_two = None

    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_show = frame.copy()

            frame_counter += 1
            use_gesture = (frame_counter % 2 == 0)

            digit, two_hands = (None, False)
            if use_gesture:
                digit, two_hands = detect_gestures(frame)

            if mode == MODE_LOCKED:
                if digit is not None and digit != prev_digit:
                    password_input.append(digit)
                    if len(password_input) == len(PASSWORD):
                        if password_input == PASSWORD:
                            mode = MODE_PHOTO
                            password_input = []
                        else:
                            password_input = []
                if digit is None:
                    prev_digit = None
                else:
                    prev_digit = digit

                cv2.putText(frame_show,
                            f"LOCKED - seq: {password_input}",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2)

            else:
                if two_hands and not prev_two_hands:
                    if mode == MODE_PHOTO:
                        mode = MODE_TWO
                        templates_two = None
                    elif mode == MODE_TWO:
                        mode = MODE_PHOTO
                prev_two_hands = two_hands

                if mode == MODE_PHOTO:
                    if current_photo is not None:
                        frame_show, prev_dst_pts_photo = aplicar_swap_foto(
                            frame, current_photo, current_mask, current_src_pts, prev_dst_pts_photo
                        )
                    cv2.putText(frame_show,
                                "MODO FOTO (c/v cambio, r recarga)",
                                (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2)

                elif mode == MODE_TWO:
                    if templates_two is None:
                        cv2.putText(frame_show,
                                    "MODO 2 PERSONAS - pulsa f para capturar caras",
                                    (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 0),
                                    2)
                    else:
                        frame_show = aplicar_swap_dos_caras(frame, templates_two)
                        cv2.putText(frame_show,
                                    "MODO 2 PERSONAS (doble mano vuelve a foto)",
                                    (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 0),
                                    2)

            cv2.imshow("FaceSwap Proyecto", frame_show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            if mode == MODE_PHOTO:
                if key == ord('c') and photo_paths:
                    photo_idx = (photo_idx + 1) % len(photo_paths)
                    current_photo, current_mask, current_src_pts = preparar_foto(photo_paths[photo_idx])
                    prev_dst_pts_photo = None
                elif key == ord('v') and photo_paths:
                    photo_idx = (photo_idx - 1) % len(photo_paths)
                    current_photo, current_mask, current_src_pts = preparar_foto(photo_paths[photo_idx])
                    prev_dst_pts_photo = None
                elif key == ord('r'):
                    photo_paths = load_photo_paths(photo_folder)
                    if photo_paths:
                        photo_idx = 0
                        current_photo, current_mask, current_src_pts = preparar_foto(photo_paths[photo_idx])
                        prev_dst_pts_photo = None

            if mode == MODE_TWO and key == ord('f'):
                templates_two = capturar_plantillas_dos_caras(frame)
                if templates_two is not None:
                    print("Plantillas capturadas para las dos caras")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
