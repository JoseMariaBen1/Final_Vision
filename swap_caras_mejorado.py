import cv2
import numpy as np

# === Detectores Haar: algoritmo de Viola-Jones ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# === ORB opcional (mejora extra) ===
orb = cv2.ORB_create(500)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


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

    if mode == "rect":
        blended = cv2.addWeighted(patch_src, alpha, patch_base, 1 - alpha, 0)
        base[y1:y2, x1:x2] = blended
        return base

    # máscara elíptica suave
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


def preprocesar_gray(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    eq = cv2.equalizeHist(blur)
    return eq


def detectar_landmarks_basicos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocesar_gray(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    roi_gray = gray[y:y + h, x:x + w]

    # ojos
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    eyes = sorted(eyes, key=lambda e: e[1])

    eye_centers = []
    for ex, ey, ew, eh in eyes:
        cx = x + ex + ew // 2
        cy = y + ey + eh // 2
        eye_centers.append((cx, cy))

    if len(eye_centers) >= 2:
        eye_centers = eye_centers[:2]
        eye_centers = sorted(eye_centers, key=lambda p: p[0])
        left_eye = np.array(eye_centers[0], dtype=np.float32)
        right_eye = np.array(eye_centers[1], dtype=np.float32)
    elif len(eye_centers) == 1:
        left_eye = np.array(eye_centers[0], dtype=np.float32)
        right_eye = np.array([x + w - (left_eye[0] - x), left_eye[1]], dtype=np.float32)
    else:
        left_eye = np.array([x + 0.3 * w, y + 0.35 * h], dtype=np.float32)
        right_eye = np.array([x + 0.7 * w, y + 0.35 * h], dtype=np.float32)

    # boca geométrica
    eye_center = (left_eye + right_eye) / 2.0
    eye_dist = np.linalg.norm(right_eye - left_eye)

    MOUTH_OFFSET_FACTOR = 1.2
    mouth_center = eye_center + np.array([0.0, MOUTH_OFFSET_FACTOR * eye_dist], dtype=np.float32)

    pts = np.vstack([left_eye, right_eye, mouth_center]).astype(np.float32)
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

    return img, mask, src_pts, face_rect


def refinar_afn_con_orb(photo_gray, frame_gray, src_face_rect, dst_face_rect, M_inicial):
    # Mejora opcional; si algo falla, devuelve M_inicial
    if M_inicial is None:
        return M_inicial

    x_s, y_s, w_s, h_s = src_face_rect
    x_d, y_d, w_d, h_d = dst_face_rect

    roi_photo = photo_gray[y_s:y_s + h_s, x_s:x_s + w_s]
    roi_frame = frame_gray[y_d:y_d + h_d, x_d:x_d + w_d]

    if roi_photo.size == 0 or roi_frame.size == 0:
        return M_inicial

    kp1, des1 = orb.detectAndCompute(roi_photo, None)
    kp2, des2 = orb.detectAndCompute(roi_frame, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return M_inicial

    matches = bf_matcher.match(des1, des2)
    if len(matches) < 8:
        return M_inicial

    matches = sorted(matches, key=lambda m: m.distance)[:40]

    pts_photo = []
    pts_frame = []
    for m in matches:
        p = kp1[m.queryIdx].pt
        q = kp2[m.trainIdx].pt
        pts_photo.append([p[0] + x_s, p[1] + y_s])
        pts_frame.append([q[0] + x_d, q[1] + y_d])

    pts_photo = np.float32(pts_photo).reshape(-1, 1, 2)
    pts_frame = np.float32(pts_frame).reshape(-1, 1, 2)

    M2, inliers = cv2.estimateAffine2D(
        pts_photo, pts_frame,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if M2 is None:
        return M_inicial

    return M2.astype(np.float32)


def aplicar_swap(frame, photo_img, photo_mask, src_pts, src_face_rect,
                 prev_dst_pts=None, alpha_smooth=0.6, usar_orb=False):
    dst_pts, dst_face_rect = detectar_landmarks_basicos(frame)
    if dst_pts is None or dst_face_rect is None:
        return frame, prev_dst_pts

    if prev_dst_pts is not None:
        dst_pts = alpha_smooth * dst_pts + (1 - alpha_smooth) * prev_dst_pts

    M = cv2.getAffineTransform(src_pts, dst_pts.astype(np.float32))

    if usar_orb:
        photo_gray = cv2.cvtColor(photo_img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = preprocesar_gray(frame_gray)
        M = refinar_afn_con_orb(photo_gray, frame_gray, src_face_rect, dst_face_rect, M)

    h, w = frame.shape[:2]
    warped_face = cv2.warpAffine(photo_img, M, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(photo_mask, M, (w, h))

    x_d, y_d, w_d, h_d = dst_face_rect
    mask_dst = np.zeros_like(warped_mask)
    center_d = (x_d + w_d // 2, y_d + h_d // 2)
    axes_d = (int(w_d * 0.5), int(h_d * 0.6))
    cv2.ellipse(mask_dst, center_d, axes_d, 0, 0, 360, 255, -1)

    mask_final = cv2.bitwise_and(warped_mask, mask_dst)

    ys, xs = np.where(mask_final > 0)
    if len(xs) == 0 or len(ys) == 0:
        return frame, prev_dst_pts

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    src_roi = warped_face[y1:y2 + 1, x1:x2 + 1]
    mask_roi = mask_final[y1:y2 + 1, x1:x2 + 1]
    center_clone = ((x1 + x2) // 2, (y1 + y2) // 2)

    swapped = cv2.seamlessClone(src_roi, frame, mask_roi, center_clone, cv2.NORMAL_CLONE)

    # overlay suave de ojos y boca para mantener expresiones
    left_eye = dst_pts[0]
    right_eye = dst_pts[1]
    mouth = dst_pts[2]

    eye_dist = np.linalg.norm(right_eye - left_eye)

    eye_w = int(eye_dist * 0.8)
    eye_h = int(eye_dist * 0.35)

    mouth_w = int(eye_dist * 1.1)
    mouth_h = int(eye_dist * 0.3)

    swapped = overlay_region(swapped, frame, left_eye, eye_w, eye_h, alpha=0.9, mode="ellipse")
    swapped = overlay_region(swapped, frame, right_eye, eye_w, eye_h, alpha=0.9, mode="ellipse")
    swapped = overlay_region(swapped, frame, mouth, mouth_w, mouth_h, alpha=0.8, mode="ellipse")

    return swapped, dst_pts
