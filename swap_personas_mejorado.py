import cv2
import numpy as np

# Detectores Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


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

    eye_center = (left_eye + right_eye) / 2.0
    eye_dist = np.linalg.norm(right_eye - left_eye)

    MOUTH_OFFSET_FACTOR = 1.2
    mouth_center = eye_center + np.array([0.0, MOUTH_OFFSET_FACTOR * eye_dist], dtype=np.float32)

    pts = np.vstack([left_eye, right_eye, mouth_center]).astype(np.float32)
    face_rect = (x, y, w, h)
    return pts, face_rect


def detectar_landmarks_multiple(frame, max_faces=2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = preprocesar_gray(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) < max_faces:
        return None, None

    faces_sorted = sorted(faces, key=lambda f: f[0])[:max_faces]

    pts_list = []
    rect_list = []

    MOUTH_OFFSET_FACTOR = 1.2

    for (x, y, w, h) in faces_sorted:
        roi_gray = gray[y:y + h, x:x + w]
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
            right_eye = np.array(
                [x + w - (left_eye[0] - x), left_eye[1]],
                dtype=np.float32,
            )
        else:
            left_eye = np.array(
                [x + 0.3 * w, y + 0.35 * h],
                dtype=np.float32,
            )
            right_eye = np.array(
                [x + 0.7 * w, y + 0.35 * h],
                dtype=np.float32,
            )

        eye_center = (left_eye + right_eye) / 2.0
        eye_dist = np.linalg.norm(right_eye - left_eye)

        mouth_center = eye_center + np.array(
            [0.0, MOUTH_OFFSET_FACTOR * eye_dist],
            dtype=np.float32,
        )
        mouth_center[1] = min(max(mouth_center[1], y + 0.5 * h), y + h - 5)

        pts = np.vstack([left_eye, right_eye, mouth_center]).astype(np.float32)
        pts_list.append(pts)
        rect_list.append((x, y, w, h))

    return pts_list, rect_list


def capturar_plantillas_dos_caras(frame):
    pts_list, rect_list = detectar_landmarks_multiple(frame, max_faces=2)
    if pts_list is None:
        return None

    templates = []
    for pts, (x, y, w, h) in zip(pts_list, rect_list):
        crop = frame[y:y + h, x:x + w].copy()
        offset = np.float32([[x, y], [x, y], [x, y]])
        src_pts_local = pts - offset

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.5), int(h * 0.6))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        templates.append((crop, mask, src_pts_local.astype(np.float32)))

    if len(templates) < 2:
        return None

    return templates


def clonar_cara(frame, src_img, src_mask, src_pts, dst_pts, dst_face_rect):
    M = cv2.getAffineTransform(src_pts, dst_pts.astype(np.float32))
    h, w = frame.shape[:2]
    warped_face = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(src_mask, M, (w, h))

    x_d, y_d, w_d, h_d = dst_face_rect
    mask_dst = np.zeros_like(warped_mask)
    center_d = (x_d + w_d // 2, y_d + h_d // 2)
    axes_d = (int(w_d * 0.5), int(h_d * 0.6))
    cv2.ellipse(mask_dst, center_d, axes_d, 0, 0, 360, 255, -1)

    mask_final = cv2.bitwise_and(warped_mask, mask_dst)

    ys, xs = np.where(mask_final > 0)
    if len(xs) == 0 or len(ys) == 0:
        return frame

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    src_roi = warped_face[y1:y2 + 1, x1:x2 + 1]
    mask_roi = mask_final[y1:y2 + 1, x1:x2 + 1]
    center_clone = ((x1 + x2) // 2, (y1 + y2) // 2)

    output = cv2.seamlessClone(src_roi, frame, mask_roi, center_clone, cv2.NORMAL_CLONE)
    return output


def aplicar_swap_dos_caras(frame, templates):
    pts_list, rect_list = detectar_landmarks_multiple(frame, max_faces=2)
    if pts_list is None:
        return frame

    (img_left, mask_left, src_pts_left) = templates[0]
    (img_right, mask_right, src_pts_right) = templates[1]

    dst_pts_left = pts_list[0]
    dst_pts_right = pts_list[1]
    rect_left = rect_list[0]
    rect_right = rect_list[1]

    out = frame.copy()
    out = clonar_cara(out, img_left, mask_left, src_pts_left, dst_pts_right, rect_right)
    out = clonar_cara(out, img_right, mask_right, src_pts_right, dst_pts_left, rect_left)

    # refinamiento ojos/boca, mismos parámetros que modo foto
    left_eye_L, right_eye_L, mouth_L = dst_pts_left
    eye_dist_L = np.linalg.norm(right_eye_L - left_eye_L)
    eye_w_L = int(eye_dist_L * 0.8)
    eye_h_L = int(eye_dist_L * 0.35)
    mouth_w_L = int(eye_dist_L * 1.1)
    mouth_h_L = int(eye_dist_L * 0.3)

    out = overlay_region(out, frame, left_eye_L, eye_w_L, eye_h_L, alpha=0.9, mode="ellipse")
    out = overlay_region(out, frame, right_eye_L, eye_w_L, eye_h_L, alpha=0.9, mode="ellipse")
    out = overlay_region(out, frame, mouth_L, mouth_w_L, mouth_h_L, alpha=0.8, mode="ellipse")

    left_eye_R, right_eye_R, mouth_R = dst_pts_right
    eye_dist_R = np.linalg.norm(right_eye_R - left_eye_R)
    eye_w_R = int(eye_dist_R * 0.8)
    eye_h_R = int(eye_dist_R * 0.35)
    mouth_w_R = int(eye_dist_R * 1.1)
    mouth_h_R = int(eye_dist_R * 0.3)

    out = overlay_region(out, frame, left_eye_R, eye_w_R, eye_h_R, alpha=0.9, mode="ellipse")
    out = overlay_region(out, frame, right_eye_R, eye_w_R, eye_h_R, alpha=0.9, mode="ellipse")
    out = overlay_region(out, frame, mouth_R, mouth_w_R, mouth_h_R, alpha=0.8, mode="ellipse")

    return out
