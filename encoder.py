import cv2
import numpy as np

PATTERN_NONE = 0
PATTERN_CIRCLE = 1
PATTERN_TRIANGLE = 2
PATTERN_SQUARE = 3


def clasificar_contorno(cnt):
    area = cv2.contourArea(cnt)
    if area < 1000:
        return PATTERN_NONE

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return PATTERN_NONE

    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    vertices = len(approx)

    circularity = 4 * np.pi * area / (peri * peri)

    if circularity > 0.8:
        return PATTERN_CIRCLE

    if vertices == 3:
        return PATTERN_TRIANGLE

    if 4 <= vertices <= 6:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h)
        if 0.8 <= ratio <= 1.2:
            return PATTERN_SQUARE

    return PATTERN_NONE


def detectar_patron(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    h, w = thresh.shape
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    y1 = int(h * 0.2)
    y2 = int(h * 0.8)
    roi = thresh[y1:y2, x1:x2]

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return PATTERN_NONE, (x1, y1, x2, y2), None

    cnt = max(contours, key=cv2.contourArea)
    cnt_shifted = cnt + np.array([[x1, y1]])

    patron = clasificar_contorno(cnt_shifted)

    return patron, (x1, y1, x2, y2), cnt_shifted


def nombre_patron(code):
    if code == PATTERN_CIRCLE:
        return "CIRCULO"
    if code == PATTERN_TRIANGLE:
        return "TRIANGULO"
    if code == PATTERN_SQUARE:
        return "CUADRADO"
    return "NINGUNO"


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_show = frame.copy()

            patron, (x1, y1, x2, y2), cnt = detectar_patron(frame)

            cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if cnt is not None:
                cv2.drawContours(frame_show, [cnt], -1, (0, 255, 0), 2)

            txt = "PATRON: " + nombre_patron(patron)
            cv2.putText(frame_show, txt, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Detector de figuras", frame_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
