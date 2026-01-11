import cv2
import numpy as np

PATTERN_NONE = 0
PATTERN_CIRCLE = 1
PATTERN_TRIANGLE = 2
PATTERN_SQUARE = 3

MIN_LEN = 4  # longitud mínima de la contraseña


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

    x1 = int(w * 0.3)
    x2 = int(w * 0.7)
    y1 = int(h * 0.3)
    y2 = int(h * 0.7)
    roi = thresh[y1:y2, x1:x2]

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return PATTERN_NONE, (x1, y1, x2, y2), None

    roi_h, roi_w = roi.shape
    min_area = 0.08 * roi_w * roi_h

    candidatos = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ratio_minmax = min(cw, ch) / float(max(cw, ch))
        if ratio_minmax < 0.35:
            continue
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        if not (roi_w * 0.25 < cx < roi_w * 0.75 and roi_h * 0.25 < cy < roi_h * 0.75):
            continue
        candidatos.append(c)

    if not candidatos:
        return PATTERN_NONE, (x1, y1, x2, y2), None

    cnt = max(candidatos, key=cv2.contourArea)
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


def secuencia_a_texto(seq):
    return "-".join(nombre_patron(p) for p in seq)


def crear_contrasena(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    seq = []

    print("MODO CREAR CONTRASENA")
    print(f"Longitud minima: {MIN_LEN}")
    print("Coloca la figura en el recuadro y pulsa 'c' para capturar.")
    print("'e' = guardar, 'r' = reset, 'q' = salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_show = frame.copy()

        patron, (x1, y1, x2, y2), cnt = detectar_patron(frame)

        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if cnt is not None:
            cv2.drawContours(frame_show, [cnt], -1, (0, 255, 0), 2)

        txt1 = "CREAR CONTRASENA"
        cv2.putText(frame_show, txt1, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        txt2 = "SEC: " + secuencia_a_texto(seq)
        cv2.putText(frame_show, txt2, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        txt3 = f"Min len: {MIN_LEN} | 'c'=captura, 'e'=guardar, 'r'=reset, 'q'=salir"
        cv2.putText(frame_show, txt3, (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        txt4 = "PATRON ACTUAL: " + nombre_patron(patron)
        cv2.putText(frame_show, txt4, (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Seguridad - Crear contrasena", frame_show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            seq = None
            break

        if key == ord('r'):
            seq = []
            print("Secuencia reseteada.")

        if key == ord('c'):
            if patron != PATTERN_NONE:
                seq.append(patron)
                print("Secuencia actual:", secuencia_a_texto(seq))
            else:
                print("No se ha detectado figura valida al capturar.")

        if key == ord('e'):
            if len(seq) >= MIN_LEN:
                print("Contrasena guardada:", secuencia_a_texto(seq))
                break
            else:
                print(f"La contrasena debe tener al menos {MIN_LEN} figuras.")

    cap.release()
    cv2.destroyAllWindows()
    return seq



def comprobar_contrasena(password, camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return False, password

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    input_seq = []
    unlocked = False

    print("MODO INTRODUCIR CONTRASENA")
    print("Coloca la figura en el recuadro y pulsa 'c' para capturar cada simbolo.")
    print("'r' = cambiar contrasena, 'q' = salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_show = frame.copy()

        patron, (x1, y1, x2, y2), cnt = detectar_patron(frame)

        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if cnt is not None:
            cv2.drawContours(frame_show, [cnt], -1, (0, 255, 0), 2)

        txt1 = "INTRODUCIR CONTRASENA"
        cv2.putText(frame_show, txt1, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        txt3 = "ENT: " + secuencia_a_texto(input_seq)
        cv2.putText(frame_show, txt3, (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        txt4 = "'c'=captura, 'r'=cambiar contrasena, 'q'=salir"
        cv2.putText(frame_show, txt4, (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        txt5 = "PATRON ACTUAL: " + nombre_patron(patron)
        cv2.putText(frame_show, txt5, (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Seguridad - Introducir contrasena", frame_show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            password = None
            break

        if key == ord('r'):
            print("Cambio de contrasena solicitado.")
            cap.release()
            cv2.destroyAllWindows()
            nueva = crear_contrasena(camera_index, width, height)
            return False, nueva

        if key == ord('c'):
            if patron == PATTERN_NONE:
                print("No se ha detectado figura valida al capturar.")
            else:
                input_seq.append(patron)
                print("Entrada actual:", secuencia_a_texto(input_seq))

                if input_seq != password[:len(input_seq)]:
                    print("Error en la secuencia. Se resetea entrada.")
                    input_seq = []
                elif len(input_seq) == len(password):
                    print("CONTRASENA CORRECTA")
                    unlocked = True
                    break

    cap.release()
    cv2.destroyAllWindows()
    return unlocked, password


def barrera_seguridad(password=None, camera_index=0, width=1280, height=720):
    if not password:
        password = crear_contrasena(camera_index, width, height)
        if not password:
            return False, None

    while True:
        ok, password = comprobar_contrasena(password, camera_index, width, height)
        if password is None:
            return False, None
        if ok:
            return True, password
