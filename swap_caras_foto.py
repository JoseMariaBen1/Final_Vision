import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


import cv2
import numpy as np

def overlay_region(base, source, center, w, h, alpha=0.7, mode="rect"):
    h_img, w_img = base.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    x1 = max(cx - w // 2, 0)
    y1 = max(cy - h // 2, 0)
    x2 = min(cx + w // 2, w_img - 1)
    y2 = min(cy + h // 2, h_img - 1)

    if x2 <= x1 or y2 <= y1:
        return base

    patch_base = base[y1:y2, x1:x2]
    patch_src = source[y1:y2, x1:x2]

    
    ph, pw = patch_base.shape[:2]
    mask = np.zeros((ph, pw), dtype=np.uint8)
    center_ellipse = (pw // 2, ph // 2)
    axes = (int(pw * 0.45), int(ph * 0.6))
    cv2.ellipse(mask, center_ellipse, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3)

    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = patch_src * (alpha * mask_f) + patch_base * (1.0 - alpha * mask_f)
    blended = blended.astype(np.uint8)

    base[y1:y2, x1:x2] = blended
    return base

def detectar_landmarks_basicos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    roi_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    eyes = sorted(eyes, key=lambda e: e[1])  # ordenar por y (de arriba hacia abajo)

    eye_centers = []
    for ex, ey, ew, eh in eyes:
        cx = x + ex + ew // 2
        cy = y + ey + eh // 2
        eye_centers.append((cx, cy))

    if len(eye_centers) >= 2:
        eye_centers = eye_centers[:2]
        eye_centers = sorted(eye_centers, key=lambda p: p[0])
        left_eye, right_eye = eye_centers[0], eye_centers[1]
    elif len(eye_centers) == 1:
        left_eye = eye_centers[0]
        right_eye = (x + w - (left_eye[0] - x), left_eye[1])
    else:
        left_eye = (x + int(0.3 * w), y + int(0.35 * h))
        right_eye = (x + int(0.7 * w), y + int(0.35 * h))

    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 15)
    mouth_center = None
    if len(mouths) > 0:
        mx, my, mw, mh = sorted(mouths, key=lambda m: m[1], reverse=True)[0]
        mcx = x + mx + mw // 2
        mcy = y + my + mh // 2
        mouth_center = (mcx, mcy)
    else:
        mcx = x + w // 2
        mcy = y + int(0.75 * h)
        mouth_center = (mcx, mcy)

    pts = np.float32([left_eye, right_eye, mouth_center])
    face_rect = (x, y, w, h)
    return pts, face_rect


def preparar_foto(photo_path):
    img = cv2.imread(photo_path)
    if img is None:
        raise ValueError("No se pudo leer la foto")

    src_pts, face_rect = detectar_landmarks_basicos(img)
    if src_pts is None:
        raise ValueError("No se detectó cara en la foto")

    x, y, w, h = face_rect
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    center = (x + w // 2, y + h // 2)
    axes = (int(w * 0.5), int(h * 0.6))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    return img, mask, src_pts


def aplicar_swap(frame, photo_img, photo_mask, src_pts, prev_dst_pts=None, alpha_smooth=0.6):
    dst_pts, _ = detectar_landmarks_basicos(frame)
    if dst_pts is None:
        return frame, prev_dst_pts

    if prev_dst_pts is not None:
        dst_pts = alpha_smooth * dst_pts + (1 - alpha_smooth) * prev_dst_pts

    M = cv2.getAffineTransform(src_pts, dst_pts.astype(np.float32))
    h, w = frame.shape[:2]
    warped_face = cv2.warpAffine(photo_img, M, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(photo_mask, M, (w, h))

    ys, xs = np.where(warped_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return frame, prev_dst_pts

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    src_roi = warped_face[y1:y2+1, x1:x2+1]
    mask_roi = warped_mask[y1:y2+1, x1:x2+1]
    center_clone = ((x1 + x2) // 2, (y1 + y2) // 2)

    swapped = cv2.seamlessClone(src_roi, frame, mask_roi, center_clone, cv2.NORMAL_CLONE)

    left_eye = dst_pts[0]
    right_eye = dst_pts[1]
    mouth = dst_pts[2]

    face_width = np.linalg.norm(dst_pts[1] - dst_pts[0])

    eye_w = int(face_width * 0.5)
    eye_h = int(face_width * 0.3)
    mouth_w = int(face_width * 0.8)
    mouth_h = int(face_width * 0.4)

    swapped = overlay_region(swapped, frame, left_eye, eye_w, eye_h, alpha=0.8)
    swapped = overlay_region(swapped, frame, right_eye, eye_w, eye_h, alpha=0.8)
    swapped = overlay_region(swapped, frame, mouth, mouth_w, mouth_h, alpha=0.9, mode='mouth')

    return swapped, dst_pts




def main(camera_index=0, width=1280, height=720, photo_path="cara_foto.jpg"):
    photo_img, photo_mask, src_pts = preparar_foto(photo_path)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    prev_dst_pts = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_swapped, prev_dst_pts = aplicar_swap(frame, photo_img, photo_mask, src_pts, prev_dst_pts)

            cv2.imshow("Face swap alineado ojos y boca", frame_swapped)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main(photo_path="cara_mariano_rajoy.webp")
