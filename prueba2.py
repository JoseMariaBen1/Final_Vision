import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#quita frames por segundo que y porque justificar

def detectar_landmarks_multiple(frame, max_faces=2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) < max_faces:
        return None, None

    faces_sorted = sorted(faces, key=lambda f: f[0])[:max_faces]

    pts_list = []
    rect_list = []

    for (x, y, w, h) in faces_sorted:
        roi_gray = gray[y:y+h, x:x+w]
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
            left_eye, right_eye = eye_centers[0], eye_centers[1]
        elif len(eye_centers) == 1:
            left_eye = eye_centers[0]
            right_eye = (x + w - (left_eye[0] - x), left_eye[1])
        else:
            left_eye = (x + int(0.3 * w), y + int(0.35 * h))
            right_eye = (x + int(0.7 * w), y + int(0.35 * h))

        mcx = x + w // 2
        mcy = y + int(0.75 * h)
        mouth_center = (mcx, mcy)

        pts = np.float32([left_eye, right_eye, mouth_center])
        pts_list.append(pts)
        rect_list.append((x, y, w, h))

    return pts_list, rect_list


def capturar_plantillas_dos_caras(frame):
    pts_list, rect_list = detectar_landmarks_multiple(frame, max_faces=2)
    if pts_list is None:
        return None

    templates = []

    for pts, (x, y, w, h) in zip(pts_list, rect_list):
        crop = frame[y:y+h, x:x+w].copy()
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


def clonar_cara(frame, src_img, src_mask, src_pts, dst_pts):
    M = cv2.getAffineTransform(src_pts, dst_pts.astype(np.float32))
    h, w = frame.shape[:2]
    warped_face = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(src_mask, M, (w, h))

    ys, xs = np.where(warped_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return frame

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    src_roi = warped_face[y1:y2+1, x1:x2+1]
    mask_roi = warped_mask[y1:y2+1, x1:x2+1]
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    output = cv2.seamlessClone(src_roi, frame, mask_roi, center, cv2.NORMAL_CLONE)
    return output


def aplicar_swap_dos_caras(frame, templates):
    pts_list, rect_list = detectar_landmarks_multiple(frame, max_faces=2)
    if pts_list is None:
        return frame

    (img_left, mask_left, src_pts_left) = templates[0]
    (img_right, mask_right, src_pts_right) = templates[1]

    dst_pts_left = pts_list[0]
    dst_pts_right = pts_list[1]

    out = frame.copy()
    out = clonar_cara(out, img_left, mask_left, src_pts_left, dst_pts_right)
    out = clonar_cara(out, img_right, mask_right, src_pts_right, dst_pts_left)

    return out


def main(camera_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    templates = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):
                templates = capturar_plantillas_dos_caras(frame)
                if templates is not None:
                    print("Plantillas capturadas para las dos caras")

            if templates is not None:
                frame_out = aplicar_swap_dos_caras(frame, templates)
            else:
                frame_out = frame

            cv2.imshow("Swap dos personas (pulsa f para capturar, q para salir)", frame_out)

            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
