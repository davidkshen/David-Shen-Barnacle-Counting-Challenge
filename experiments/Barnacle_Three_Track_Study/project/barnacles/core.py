"""Geometry, reference-label extraction, fixed convolutions, and object detection.

The detector NEVER reads annotation masks. Only load_reference() does, for
supervision/evaluation. Coordinates below refer to the rectified image.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Config:
    size: int = 1024
    radii: tuple = (3., 4., 5.5, 7.5, 10., 13., 17.)
    proposal_floor: float = 0.12
    proposal_spacing: float = 3.
    seed: int = 42


def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def rectify(rgb, corners, size=1024):
    """Four HUMAN-CONFIRMED inside corners, clockwise from top left.

    This normalizes frame size/perspective; it does not recover rock topography.
    """
    src = np.asarray(corners, np.float32)
    if src.shape != (4, 2) or not cv2.isContourConvex(src.astype(np.int32)):
        raise ValueError('Supply four convex corners: TL, TR, BR, BL.')
    if cv2.contourArea(src) < 100:
        raise ValueError('Frame polygon is too small.')
    h, w = rgb.shape[:2]
    if np.any(src < 0) or np.any(src[:, 0] >= w) or np.any(src[:, 1] >= h):
        raise ValueError('Frame corners must lie within the input photograph.')
    dst = np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgb, matrix, (size, size)), matrix


def load_reference(path, matrix, size=1024, alpha=32, min_area=6):
    """Recover enclosed transparent holes in the blue contour PNG.

    A contour is a stroke, not a filled instance. Counting stroke components
    merges touching outlines; instead we count enclosed background regions.
    The alpha/area cutoffs are explicit, imperfect label-decoding assumptions.
    They are held fixed and sensitivity is reported separately.
    """
    a = np.asarray(Image.open(path).convert('RGBA'))
    blue = (a[..., 2].astype(float) > a[..., 0]+40.) & (a[..., 2] > a[..., 1])
    stroke = blue & (a[..., 3] >= alpha)
    holes = ndi.binary_fill_holes(stroke) & ~stroke
    lab, n = ndi.label(holes)
    areas = np.bincount(lab.ravel())
    ids = np.flatnonzero(areas >= min_area)
    ids = ids[ids != 0]
    if not len(ids):
        raise ValueError(f'No closed labels recovered from {path}')
    centers = np.array(ndi.center_of_mass(holes, lab, ids))[:, ::-1]
    # An effective aperture radius includes approximately half the stroke width.
    radii = np.sqrt(areas[ids]/np.pi) + 1.5
    xy = cv2.perspectiveTransform(centers[:, None, :].astype(np.float32), matrix)[:, 0]
    offsets = centers.copy(); offsets[:, 0] += radii
    xo = cv2.perspectiveTransform(offsets[:, None, :].astype(np.float32), matrix)[:, 0]
    rr = np.linalg.norm(xo-xy, axis=1)
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
    return np.column_stack((xy[keep], rr[keep])).astype(np.float32)


def normalized_gray(rgb):
    """Local contrast normalization on log luminance; not a height estimate.

    Finite filter support is <=48 pixels (sigma=12, truncate=4).
    It attenuates smooth illumination changes but cannot restore clipped detail.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    g = np.log(g + 0.04)
    mean = ndi.gaussian_filter(g, 12, truncate=4)
    residual = g - mean
    rms = np.sqrt(ndi.gaussian_filter(residual**2, 12, truncate=4) + .025**2)
    return np.clip(residual / rms, -4, 4).astype(np.float32)


def ring_kernel(radius):
    """Positive annulus minus negative disk. Zero sum rejects uniform offsets."""
    n = int(np.ceil(radius*1.5))
    y, x = np.mgrid[-n:n+1, -n:n+1]
    d = np.hypot(x, y)
    ring = ((d >= .8*radius) & (d <= 1.35*radius)).astype(np.float32)
    disk = (d < .65*radius).astype(np.float32)
    return ring/ring.sum() - disk/disk.sum()


def suppress(points, scores, distance=3., radii=None):
    """Greedy score-ordered nonmaximum suppression with spatial bins."""
    if len(points) == 0:
        return np.array([], dtype=int)
    cell = max(distance, 2.)
    bins = {}; keep = []
    for i in np.argsort(-scores, kind='stable'):
        x, y = points[i]
        key = (int(x//cell), int(y//cell))
        reach = int(np.ceil(max(distance, 1.2*(radii[i] if radii is not None else 0))/cell))
        reject = False
        for dx in range(-reach, reach+1):
            for dy in range(-reach, reach+1):
                for j in bins.get((key[0]+dx, key[1]+dy), []):
                    limit = distance if radii is None else max(distance, .7*min(radii[i], radii[j]))
                    if np.sum((points[i]-points[j])**2) < limit**2:
                        reject = True; break
                if reject: break
            if reject: break
        if not reject:
            keep.append(i); bins.setdefault(key, []).append(i)
    return np.array(keep, dtype=int)


def propose(rgb, config=Config()):
    """High-recall proposal stage: dark-center/bright-rim convolutions.

    This deliberately admits rock pits and empty shells. The classifier and
    human audit have to deal with them. The output is x,y,radius,ring_response.
    """
    g = normalized_gray(rgb)
    rows = []
    for r in config.radii:
        response = cv2.filter2D(g, -1, ring_kernel(r), borderType=cv2.BORDER_REFLECT101)
        peak = response == ndi.maximum_filter(response, size=5)
        yy, xx = np.where(peak & (response > config.proposal_floor))
        rows.append(np.column_stack((xx, yy, np.full(len(xx), r), response[yy, xx])))
    all_points = np.concatenate(rows).astype(np.float32)
    ids = suppress(all_points[:, :2], all_points[:, 3], config.proposal_spacing)
    return all_points[ids], g


def sample(array, x, y):
    return ndi.map_coordinates(array, [np.asarray(y).ravel(), np.asarray(x).ravel()],
                               order=1, mode='reflect').reshape(np.asarray(x).shape)


def features(gray, proposals, include_slit=True):
    """Rotation-pooled local features; no global coordinates, color or density.

    Three relative scales soften the proposal radius decision. Circular sampling
    and max over line orientations avoid needing isolated barnacle cutouts.
    Maximum sampling offset is 1.5 * 1.6 * 17 = 40.8 pixels.
    """
    n = len(proposals)
    if n == 0:
        raise ValueError('No proposals: try a lower proposal floor or inspect the image.')
    x = proposals[:, 0, None]; y = proposals[:, 1, None]
    theta = np.linspace(0, 2*np.pi, 32, endpoint=False)[None, :]
    out = []; names = []
    for mult in (.75, 1., 1.5):
        r = proposals[:, 2, None]*mult
        ring_values = []
        for frac in (.25, .5, .8, 1.1, 1.6):
            v = sample(gray, x+r*frac*np.cos(theta), y+r*frac*np.sin(theta))
            ring_values.append(v)
            for label, val in [('mean', v.mean(1)), ('sd', v.std(1)),
                               ('q25', np.quantile(v, .25, axis=1)),
                               ('q75', np.quantile(v, .75, axis=1))]:
                out.append(val); names.append(f'{mult:g}r_ring{frac:g}_{label}')
        inner, rim = ring_values[1], ring_values[3]
        out.extend([(rim-inner).mean(1), (rim>inner).mean(1)])
        names.extend([f'{mult:g}r_ring_contrast', f'{mult:g}r_ring_coverage'])
        # Rotation-invariant magnitudes of angular Fourier harmonics (1..4).
        harmonics = np.abs(np.fft.rfft(ring_values[2], axis=1))/32
        for k in range(1, 5):
            out.append(harmonics[:, k]); names.append(f'{mult:g}r_harmonic{k}')
        if include_slit:
            line_scores = []; coverages = []
            t = np.linspace(-.65, .65, 11)[None, :]
            for angle in np.linspace(0, np.pi, 12, endpoint=False):
                ux, uy = np.cos(angle), np.sin(angle)
                cx, cy = x+r*t*ux, y+r*t*uy
                center = sample(gray, cx, cy)
                flank1 = sample(gray, cx-r*.28*uy, cy+r*.28*ux)
                flank2 = sample(gray, cx+r*.28*uy, cy-r*.28*ux)
                delta = .5*(flank1+flank2)-center
                line_scores.append(delta.mean(1))
                coverages.append((delta>.1).mean(1))
            ls = np.stack(line_scores, 1); cs = np.stack(coverages, 1)
            best = ls.argmax(1)
            for label, val in [('slit_max', ls.max(1)), ('slit_mean', ls.mean(1)),
                               ('slit_directionality', ls.max(1)-ls.mean(1)),
                               ('slit_coverage', cs[np.arange(n), best])]:
                out.append(val); names.append(f'{mult:g}r_{label}')
    return np.stack(out, 1).astype(np.float32), names


def gradient_patches(gray, proposals, batch_size=1024):
    """A small fixed spatial gradient descriptor, without a pretrained network.

    4x4 cells, 8 unsigned orientation bins. Fixed 32-pixel field of view; unlike
    ring averages this retains WHERE edges fall inside an aperture. Sampling
    at 16x16 positions keeps the prototype inexpensive. Rotations of this
    descriptor are used for source-only augmentation in the optional experiment.
    """
    gx = ndi.sobel(gray, axis=1)/8; gy = ndi.sobel(gray, axis=0)/8
    offsets = np.linspace(-15, 15, 16)
    dy, dx = np.meshgrid(offsets, offsets, indexing='ij')
    result = []
    for start in range(0, len(proposals), batch_size):
        p = proposals[start:start+batch_size]
        x = p[:, 0, None, None]+dx; y = p[:, 1, None, None]+dy
        px = sample(gx, x, y); py = sample(gy, x, y)
        mag = np.hypot(px, py); angle = np.mod(np.arctan2(py, px), np.pi)*8/np.pi
        bins = []
        for b in range(8):
            delta = np.minimum(np.abs(angle-b), 8-np.abs(angle-b))
            vote = mag*np.maximum(1-delta, 0)
            cells = vote.reshape(len(p), 4, 4, 4, 4).sum(axis=(2, 4))
            bins.append(cells)
        h = np.stack(bins, -1)
        h /= np.sqrt((h*h).sum(-1, keepdims=True)+.1)
        result.append(h.reshape(len(p), -1))
    return np.concatenate(result).astype(np.float32)


def rotate_gradient_descriptor(H, quarter_turns):
    h = H.reshape(-1, 4, 4, 8)
    h = np.rot90(h, quarter_turns, axes=(1, 2))
    h = np.roll(h, -4*quarter_turns, axis=3)
    return h.reshape(-1, 128)


def reference_targets(proposals, reference):
    """Local supervision; ambiguous near-boundary candidates are ignored.

    Positive: within max(2,0.45r) of a labeled aperture center.
    Negative: farther than max(4,1.3r). Others: -1 (do not train on them).
    Missing annotations can still contaminate negatives; see report.
    """
    d, j = cKDTree(reference[:, :2]).query(proposals[:, :2])
    rr = reference[j, 2]
    target = np.full(len(proposals), -1, np.int8)
    target[d <= np.maximum(2., .45*rr)] = 1
    target[d > np.maximum(4., 1.3*rr)] = 0
    return target


def match_objects(predictions, reference, tolerance=.75):
    """One-to-one maximum-cardinality matching, then minimum distance.

    A large reward for any valid pair makes matching maximize the number of
    matches before minimizing normalized distance. Duplicate detections become
    false positives. This is center detection evaluation, NOT segmentation IoU.
    """
    p, g = len(predictions), len(reference)
    if not p or not g:
        return np.empty((0, 2), int)
    dist = np.linalg.norm(predictions[:, None, :2]-reference[None, :, :2], axis=2)
    limit = np.maximum(3., tolerance*reference[:, 2])
    normalized = dist/limit[None, :]
    valid = normalized <= 1.
    reward = min(p, g)+1
    cost = np.where(valid, normalized-reward, 0.)
    i, j = linear_sum_assignment(cost)
    keep = valid[i, j]
    return np.column_stack((i[keep], j[keep]))


def metrics(predictions, reference, tolerance=.75):
    matches = match_objects(predictions, reference, tolerance)
    tp = len(matches); fp = len(predictions)-tp; fn = len(reference)-tp
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(reference_count=len(reference), predicted_count=len(predictions),
                tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                signed_error=len(predictions)-len(reference),
                relative_error=(len(predictions)-len(reference))/max(1, len(reference)))


def select_detections(proposals, scores, threshold):
    ids = np.flatnonzero(scores >= threshold)
    keep = suppress(proposals[ids, :2], scores[ids], distance=4., radii=proposals[ids, 2])
    return ids[keep]
