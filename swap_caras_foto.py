import cv2
import numpy as np

# Cargamos el clasificador de caras frontal de OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def load_face_from_photo(photo_path):
    img = cv2.imread(photo_path)
    if img is None:
        raise ValueError("No se pudo leer la imagen de foto de cara")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No se detectó ninguna cara en la foto")

    # Cogemos la cara más grande (por si acaso hay varias)
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

    face_roi = img[y:y+h, x:x+w]

    # Máscara elíptica del tamaño de la cara
    mask = np.zeros(face_roi.shape[:2], dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)  # radio en x, radio en y
    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    return face_roi, mask


def paste_photo_face_on_frame(frame, x, y, w, h, photo_face, photo_mask):
    # Redimensionar la cara de la foto al tamaño de la cara detectada
    resized_face = cv2.resize(photo_face, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(photo_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Creamos una máscara 3 canales opcionalmente (no es obligatorio para seamlessClone,
    # pero viene bien si queréis visualizarla)
    # mask_3ch = cv2.merge([resized_mask, resized_mask, resized_mask])

    # Definir el centro donde clonar (centro de la cara del usuario)
    center = (x + w // 2, y + h // 2)

    # Para seamlessClone, la máscara debe tener mismo tamaño que src (aquí resized_face)
    # y ser de 1 canal (uint8), lo que ya tenemos en resized_mask.
    output = cv2.seamlessClone(
        resized_face,      # src: cara de la foto redimensionada
        frame,             # dst: frame de la webcam
        resized_mask,      # mask: máscara de la cara
        center,            # centro donde clonar
        cv2.NORMAL_CLONE   # modo de clonación
    )

    return output

def main(camera_index=0, width=1280, height=720, photo_path="cara_foto.jpg"):
    # Cargamos la cara de la foto una sola vez
    photo_face, photo_mask = load_face_from_photo(photo_path)

    # Abrimos la cámara (usa CAP_DSHOW si estás en Windows)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Si detectamos al menos una cara, cogemos la más grande
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                # Pegar cara de foto
                frame = paste_photo_face_on_frame(frame, x, y, w, h, photo_face, photo_mask)
                # (Opcional) dibujar el rectángulo de la cara detectada:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Cara ↔ Foto", frame)

            key = cv2.waitKey(1) & 0xFF
            # Pulsando 'q' o ESC se sale
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=0, width=1280, height=720, photo_path="cara_mariano_rajoy.webp")
