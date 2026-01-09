import cv2
import glob
import os

from swap_caras_mejorado import preparar_foto


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


def make_thumb(path, size=(120, 120)):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, size)


def actualizar_fotos_y_thumbs(photo_paths, photo_idx, thumb_size=(120, 120)):
    ruta_actual, photo_img, photo_mask, src_pts, src_face_rect = cargar_foto(photo_paths, photo_idx)

    n = len(photo_paths)
    prev_idx = (photo_idx - 1) % n
    next_idx = (photo_idx + 1) % n

    thumb_actual = cv2.resize(photo_img, thumb_size)
    thumb_prev = make_thumb(photo_paths[prev_idx], thumb_size)
    thumb_next = make_thumb(photo_paths[next_idx], thumb_size)

    return (ruta_actual, photo_img, photo_mask, src_pts, src_face_rect,
            thumb_prev, thumb_actual, thumb_next)
