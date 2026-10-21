import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def read_marks(marked_path, clean_path, diff=60):
    # I marked barnacles by drawing red lines on a copy of the photo.
    # compare marked vs clean: pixels that changed a lot = my marks. each connected blob = one barnacle.
    a = np.asarray(Image.open(marked_path).convert('RGB')).astype(int)
    b = np.asarray(Image.open(clean_path).convert('RGB')).astype(int)
    lab, _ = ndi.label(np.abs(a - b).sum(-1) > diff, np.ones((3, 3)))
    pts = [np.argwhere(lab[sl] == i).mean(0) + [sl[0].start, sl[1].start]
           for i, sl in enumerate(ndi.find_objects(lab), 1)]
    return np.array(pts)[:, ::-1].astype(np.float32)  # x, y


def same_barnacle_tolerance(marks, cap=15.):
    # how close a dot has to be to a mark to count as 'on that barnacle':
    # 15px, or halfway to the next mark if barnacles are packed tighter than that
    nn = cKDTree(marks).query(marks, 2)[0][:, 1]
    return np.minimum(cap, 0.5 * nn)


def find_crop(crop_path, full_path, seed=0):
    # unseen_img2 turned out to be a zoomed-in crop of unseen_img1. this finds exactly where:
    # SIFT keypoints in both, match them, fit scale + rotation + shift (RANSAC throws out bad matches).
    cv2.setRNGSeed(seed)
    a = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    src = np.float32([ka[g.queryIdx].pt for g in good])
    dst = np.float32([kb[g.trainIdx].pt for g in good])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    ok = inl.ravel() == 1
    resid = np.linalg.norm(src[ok] @ A[:, :2].T + A[:, 2] - dst[ok], axis=1)
    return A, dict(matches=len(good), inliers=int(ok.sum()), median_residual_px=float(np.median(resid)),
                   scale=float(np.hypot(A[0, 0], A[1, 0])))


def crop_to_frame(xy, A, rect_matrix):
    # crop pixel -> full photo pixel (A) -> straightened frame pixel (rect_matrix)
    full = (np.asarray(xy, np.float32) @ A[:, :2].T + A[:, 2]).astype(np.float32)
    return cv2.perspectiveTransform(full.reshape(-1, 1, 2), rect_matrix).reshape(-1, 2).astype(np.float32)


def inside(poly, xy):
    # which points are inside a polygon (the crop's outline in the frame)
    poly = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in xy], bool)


def area_scale(area, crop_size=500):
    # frame pixels per crop pixel (the crop is blown up, so this is about 0.65)
    return float(np.sqrt(cv2.contourArea(np.asarray(area, np.float32)) / (crop_size - 1.0) ** 2))
