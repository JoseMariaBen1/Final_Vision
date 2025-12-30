import cv2
import glob
import os

from swap_caras_mejorado import preparar_foto, aplicar_swap


def load_photo_paths(folder):
    paths = []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    return paths


def cargar_foto(photo_paths, idx):
    ruta = photo_paths[idx]
    photo_img, photo_mask, src_pts, src_face_rect = preparar_foto(ruta)
    return ruta, photo_img, photo_mask, src_pts, src_face_rect


def modo_fotos(camera_index=0, width=1280, height=720, photo_folder="fotos"):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    photo_paths = load_photo_paths(photo_folder)
    if not photo_paths:
        print("No se han encontrado fotos en la carpeta:", photo_folder)
        cap.release()
        return

    photo_idx = 0
    ruta_actual, photo_img, photo_mask, src_pts, src_face_rect = cargar_foto(photo_paths, photo_idx)
    prev_dst_pts = None

    print("Modo fotos:")
    print("  n = siguiente foto")
    print("  p = foto anterior")
    print("  r = recargar carpeta")
    print("  q / ESC = salir")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_swapped, prev_dst_pts = aplicar_swap(
                frame, photo_img, photo_mask, src_pts, src_face_rect,
                prev_dst_pts=prev_dst_pts, alpha_smooth=0.6, usar_orb=False
            )

            # Thumbnail de la foto actual en la esquina
            thumb_h = 160
            thumb_w = 160
            thumb = cv2.resize(photo_img, (thumb_w, thumb_h))
            frame_swapped[10:10 + thumb_h, 10:10 + thumb_w] = thumb

            nombre = os.path.basename(ruta_actual)
            cv2.putText(frame_swapped,
                        f"FOTO: {nombre}",
                        (10, 10 + thumb_h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

            cv2.putText(frame_swapped,
                        "n: siguiente | p: anterior | r: recargar | q: salir",
                        (10, frame_swapped.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

            cv2.imshow("FaceSwap - Modo fotos", frame_swapped)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

            if key == ord('n'):
                photo_idx = (photo_idx + 1) % len(photo_paths)
                ruta_actual, photo_img, photo_mask, src_pts, src_face_rect = cargar_foto(photo_paths, photo_idx)
                prev_dst_pts = None

            if key == ord('p'):
                photo_idx = (photo_idx - 1) % len(photo_paths)
                ruta_actual, photo_img, photo_mask, src_pts, src_face_rect = cargar_foto(photo_paths, photo_idx)
                prev_dst_pts = None

            if key == ord('r'):
                photo_paths = load_photo_paths(photo_folder)
                if photo_paths:
                    photo_idx = 0
                    ruta_actual, photo_img, photo_mask, src_pts, src_face_rect = cargar_foto(photo_paths, photo_idx)
                    prev_dst_pts = None
                    print("Carpeta recargada, fotos encontradas:", len(photo_paths))
                else:
                    print("Despues de recargar no hay fotos en la carpeta.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    modo_fotos()
