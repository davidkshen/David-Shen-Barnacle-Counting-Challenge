import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load_reference(mask_path, matrix, size=1024, alpha=32, min_area=6):
    # the DALI masks are blue outlines, not filled shapes.
    # so: find the blue strokes, fill them in, and keep only the holes -> one hole = one barnacle opening.
    # (counting stroke blobs instead would merge barnacles whose outlines touch)
    a = np.asarray(Image.open(mask_path).convert('RGBA'))
    blue = (a[..., 2].astype(float) > a[..., 0] + 40.) & (a[..., 2] > a[..., 1])
    stroke = blue & (a[..., 3] >= alpha)
    holes = ndi.binary_fill_holes(stroke) & ~stroke
    lab, _ = ndi.label(holes)
    areas = np.bincount(lab.ravel())
    ids = np.flatnonzero(areas >= min_area)  # drop tiny specks
    ids = ids[ids != 0]
    if not len(ids):
        raise ValueError(f'no closed outlines found in {mask_path}')
    centers = np.array(ndi.center_of_mass(holes, lab, ids))[:, ::-1]
    # radius of a circle with the same area, + half the stroke width
    radii = np.sqrt(areas[ids] / np.pi) + 1.5
    # move centres (and radii) into the straightened frame
    xy = cv2.perspectiveTransform(centers[:, None, :].astype(np.float32), matrix)[:, 0]
    off = centers.copy()
    off[:, 0] += radii
    xo = cv2.perspectiveTransform(off[:, None, :].astype(np.float32), matrix)[:, 0]
    rr = np.linalg.norm(xo - xy, axis=1)
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)  # only inside the frame
    return np.column_stack((xy[keep], rr[keep])).astype(np.float32)  # columns: x, y, radius


def in_box(xy, box):
    # which points fall inside box = (x0, y0, x1, y1)
    return (xy[:, 0] >= box[0]) & (xy[:, 0] < box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] < box[3])
