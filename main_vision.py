import cv2
import os

from security import (
    barrera_seguridad,
    detectar_patron,
    PATTERN_NONE,
    PATTERN_CIRCLE,
    PATTERN_TRIANGLE,
)
from modo_fotos import load_photo_paths, actualizar_fotos_y_thumbs
from swap_caras_mejorado import aplicar_swap
from swap_personas_mejorado import capturar_plantillas_dos_caras, aplicar_swap_dos_caras

MODE_FOTO = 0
MODE_TWO = 1


def app_vision(camera_index=0, width=1280, height=720, photo_folder="fotos"):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ----------------- Preparar fotos (modo FOTO) -----------------
    photo_paths = load_photo_paths(photo_folder)
    tiene_fotos = len(photo_paths) > 0

    photo_idx = 0
    photo_img = None
    photo_mask = None
    src_pts = None
    src_face_rect = None
    thumb_prev = None
    thumb_actual = None
    thumb_next = None
    prev_dst_pts = None

    if tiene_fotos:
        (ruta_actual, photo_img, photo_mask, src_pts, src_face_rect,
         thumb_prev, thumb_actual, thumb_next) = actualizar_fotos_y_thumbs(photo_paths, photo_idx)
        total_fotos = len(photo_paths)
    else:
        total_fotos = 0
        print("Aviso: no se han encontrado fotos en la carpeta. El modo FOTO no estara disponible.")

    # ----------------- Estado de la app -----------------
    mode = MODE_FOTO if tiene_fotos else MODE_TWO
    templates_two = None               # plantillas para modo 2 personas
    prev_pattern = PATTERN_NONE        # para detectar flanco de CÍRCULO / TRIÁNGULO

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_show = frame.copy()

            # ========== ROI LATERAL DERECHO PEQUEÑO ==========
            # Franja vertical en el lateral derecho (por ejemplo, cuarto central en altura)
            y1 = h // 4
            y2 = 3 * h // 4
            x1 = int(0.7 * w)
            x2 = w

            roi = frame[y1:y2, x1:x2]

            # Detectar figura SOLO en ese ROI
            # (detecta sobre la imagen recortada; las coords que devuelve no las usamos)
            patron, _, _ = detectar_patron(roi)

            # ---------- Lógica de cambio de modo y salida ----------
            # CÍRCULO (flanco) -> cambio de modo
            if patron == PATTERN_CIRCLE and prev_pattern != PATTERN_CIRCLE:
                if mode == MODE_FOTO and tiene_fotos:
                    mode = MODE_TWO
                    templates_two = None
                    print("Cambio a MODO 2 PERSONAS (circulo en lateral).")
                elif mode == MODE_TWO and tiene_fotos:
                    mode = MODE_FOTO
                    prev_dst_pts = None
                    print("Cambio a MODO FOTO (circulo en lateral).")

            # TRIÁNGULO (flanco) -> salir
            if patron == PATTERN_TRIANGLE and prev_pattern != PATTERN_TRIANGLE:
                print("Triangulo detectado en lateral: saliendo de la aplicacion.")
                break

            prev_pattern = patron

            # Teclas de emergencia
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Salida por teclado.")
                break

            # ================= MODO FOTO =================
            if mode == MODE_FOTO and tiene_fotos:
                # aplicar swap con la foto actual
                frame_swapped, prev_dst_pts = aplicar_swap(
                    frame, photo_img, photo_mask, src_pts, src_face_rect,
                    prev_dst_pts=prev_dst_pts, alpha_smooth=0.6, usar_orb=False
                )

                h2, w2 = frame_swapped.shape[:2]
                th, tw = thumb_actual.shape[:2]
                margin = 10

                x_prev_thumb = margin
                x_actual_thumb = x_prev_thumb + tw + margin
                x_next_thumb = x_actual_thumb + tw + margin
                y_top = margin

                # thumbnails (previa, actual, siguiente)
                if thumb_prev is not None:
                    frame_swapped[y_top:y_top + th, x_prev_thumb:x_prev_thumb + tw] = thumb_prev
                frame_swapped[y_top:y_top + th, x_actual_thumb:x_actual_thumb + tw] = thumb_actual
                if thumb_next is not None:
                    frame_swapped[y_top:y_top + th, x_next_thumb:x_next_thumb + tw] = thumb_next

                nombre = os.path.basename(photo_paths[photo_idx])
                cv2.putText(frame_swapped,
                            f"{photo_idx + 1} / {total_fotos} | {nombre}",
                            (margin, y_top + th + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2)

                cv2.putText(
                    frame_swapped,
                    "Modo FOTO | n/p: cambiar foto | r: recargar | CIRCULO lateral: modo 2 | TRIANGULO lateral: salir",
                    (margin, h2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                frame_show = frame_swapped

                # navegación de fotos
                if key == ord('n'):
                    photo_idx = (photo_idx + 1) % total_fotos
                    (ruta_actual, photo_img, photo_mask, src_pts, src_face_rect,
                     thumb_prev, thumb_actual, thumb_next) = actualizar_fotos_y_thumbs(photo_paths, photo_idx)
                    prev_dst_pts = None

                if key == ord('p'):
                    photo_idx = (photo_idx - 1) % total_fotos
                    (ruta_actual, photo_img, photo_mask, src_pts, src_face_rect,
                     thumb_prev, thumb_actual, thumb_next) = actualizar_fotos_y_thumbs(photo_paths, photo_idx)
                    prev_dst_pts = None

                if key == ord('r'):
                    photo_paths = load_photo_paths(photo_folder)
                    total_fotos = len(photo_paths)
                    if total_fotos > 0:
                        photo_idx = 0
                        (ruta_actual, photo_img, photo_mask, src_pts, src_face_rect,
                         thumb_prev, thumb_actual, thumb_next) = actualizar_fotos_y_thumbs(photo_paths, photo_idx)
                        prev_dst_pts = None
                        print("Carpeta recargada, fotos encontradas:", total_fotos)
                    else:
                        print("Despues de recargar no hay fotos en la carpeta.")
                        tiene_fotos = False
                        mode = MODE_TWO
                        templates_two = None

            # ================= MODO 2 PERSONAS =================
            elif mode == MODE_TWO:
                if templates_two is None:
                    cv2.putText(
                        frame_show,
                        "Modo 2 PERSONAS | coloca 2 caras y pulsa 'f' | CIRCULO lateral: modo foto | TRIANGULO lateral: salir",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                else:
                    frame_out = aplicar_swap_dos_caras(frame, templates_two)
                    cv2.putText(
                        frame_out,
                        "Modo 2 PERSONAS | 'f': recapturar | CIRCULO lateral: modo foto | TRIANGULO lateral: salir",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )
                    frame_show = frame_out

                if key == ord('f'):
                    templates_two = capturar_plantillas_dos_caras(frame)
                    if templates_two is not None:
                        print("Plantillas capturadas para las dos caras")

            # ========== CÍRCULO GUÍA EN EL LATERAL ==========
            # Solo dibujamos un círculo (no el rectángulo entero) para indicar la zona de dibujo
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = min((x2 - x1), (y2 - y1)) // 2 - 5
            cv2.circle(frame_show, (cx, cy), radius, (0, 255, 0), 2)
            cv2.putText(
                frame_show,
                "Dibuja CIRCULO / TRIANGULO aqui",
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.imshow("Proyecto Vision", frame_show)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # password: secuencia de figuras (por ejemplo: CUADRADO, CIRCULO, TRIANGULO, CUADRADO)
    password = [3, 1, 2, 3]
    ok, pwd = barrera_seguridad(password=password)
    if ok:
        app_vision()
    else:
        print("No se ha pasado la barrera de seguridad.")
