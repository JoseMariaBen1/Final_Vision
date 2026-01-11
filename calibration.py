import cv2
from typing import List
import numpy as np
import imageio
import copy
import os

intrinsics = None
dist_coeffs = None


def load_images(filenames: List[str]) -> List[np.ndarray]:
    return [imageio.imread(filename) for filename in filenames]


def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def write_image(output_folder: str, img_name: str, img: np.array):
    os.makedirs(output_folder, exist_ok=True)
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


def get_chessboard_points(chessboard_shape, dx, dy):
    eje_x, eje_y = chessboard_shape
    puntos = []
    for y in range(eje_y):
        for x in range(eje_x):
            coordenadas = [x * dx, y * dy, 0]
            puntos.append(coordenadas)
    return np.array(puntos, dtype=np.float32)


def calibrar_camara(
    calib_folder: str = "Calibration_images_Ray",
    n_images: int = 11,
    chessboard_shape=(9, 6),
    dx: float = 30.0,
    dy: float = 30.0,
    mostrar_esquinas: bool = False,
):
    global intrinsics, dist_coeffs

    imgs_path = [os.path.join(calib_folder, f"{i}.jpg") for i in range(1, n_images + 1)]
    imgs = load_images(imgs_path)

    corners_img = []
    for img in imgs:
        corners = cv2.findChessboardCorners(img, patternSize=chessboard_shape)
        corners_img.append(corners)

    corners_copies = []
    for corners in corners_img:
        corners_copy = copy.deepcopy(corners)
        corners_copies.append(corners_copy)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    imgs_gray = []
    for img in imgs:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs_gray.append(img_gray)

    corners_refined = [
        cv2.cornerSubPix(i, cor[1], (8, 6), (-1, -1), criteria) if cor[0] else []
        for i, cor in zip(imgs_gray, corners_copies)
    ]

    valid_imgs = []
    for img, cor in zip(imgs, corners_refined):
        if len(cor) > 0:
            cv2.drawChessboardCorners(
                img, patternSize=chessboard_shape, corners=cor, patternWasFound=True
            )
            valid_imgs.append(img)

    if mostrar_esquinas:
        for img in valid_imgs:
            show_image(img, "Esquinas detectadas")

    chessboard_points = get_chessboard_points(chessboard_shape, dx, dy)

    valid_corners = [cor[1] for cor in corners_copies if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    object_points = [chessboard_points for _ in range(len(valid_corners))]

    h, w = imgs_gray[0].shape[:2]

    rms, intr, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, valid_corners, (w, h), None, None
    )

    intrinsics = intr
    dist_coeffs = dist

    extrinsics = [
        np.hstack((cv2.Rodrigues(rvec)[0], tvec)) for rvec, tvec in zip(rvecs, tvecs)
    ]

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)

    return intrinsics, dist_coeffs, rms, extrinsics


def undistort_frame(frame: np.ndarray) -> np.ndarray:
    global intrinsics, dist_coeffs

    if intrinsics is None or dist_coeffs is None:
        raise ValueError("Camara no calibrada. Llama primero a calibrar_camara().")

    return cv2.undistort(frame, intrinsics, dist_coeffs)
