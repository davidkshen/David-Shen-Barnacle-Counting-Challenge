import cv2
import numpy as np
from PIL import Image


def read_rgb(path):
    # load any image as plain RGB (drops the alpha channel the DALI pngs have)
    return np.asarray(Image.open(path).convert('RGB'))


def rectify(rgb, corners, size=1024):
    # corners = the 4 inside corners of the green frame, clockwise from top left.
    # warp that square into a flat size x size image. same frame -> same scale on every photo,
    # which is the whole reason the frame helps (barnacle size in pixels stays about the same).
    src = np.asarray(corners, np.float32)
    if src.shape != (4, 2) or not cv2.isContourConvex(src.astype(np.int32)):
        raise ValueError('need 4 convex corners: TL, TR, BR, BL')
    h, w = rgb.shape[:2]
    if np.any(src < 0) or np.any(src[:, 0] >= w) or np.any(src[:, 1] >= h):
        raise ValueError('corners have to be inside the photo')
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgb, matrix, (size, size)), matrix
