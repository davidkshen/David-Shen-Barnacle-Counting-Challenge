# My first idea, kept for the notebook demo: describe a barnacle by hand as a bright ring around a darker middle.
# Not used by the final pipeline. The full versions of these experiments are in experiments/.
import cv2
import numpy as np
from scipy import ndimage as ndi


def normalized_gray(rgb):
    # grayscale, then divide out the slow lighting changes (sigma 12) so shadows/glare matter less
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    g = np.log(g + 0.04)
    mean = ndi.gaussian_filter(g, 12, truncate=4)
    res = g - mean
    rms = np.sqrt(ndi.gaussian_filter(res ** 2, 12, truncate=4) + .025 ** 2)
    return np.clip(res / rms, -4, 4).astype(np.float32)


def ring_kernel(radius):
    # ring (positive) minus the middle disk (negative). sums to 0, so flat areas score 0.
    n = int(np.ceil(radius * 1.5))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(x, y)
    ring = ((d >= .8 * radius) & (d <= 1.35 * radius)).astype(np.float32)
    disk = (d < .65 * radius).astype(np.float32)
    return ring / ring.sum() - disk / disk.sum()


def ring_response(gray, radii=(4, 6, 8, 11)):
    # best ring score over a few sizes at every pixel
    return np.max([cv2.filter2D(gray, -1, ring_kernel(r)) for r in radii], axis=0)


def peaks(resp, spacing=5, top=200):
    # local maxima of the response = 'this looks like a barnacle' spots
    mx = ndi.maximum_filter(resp, size=2 * spacing + 1)
    yy, xx = np.nonzero((resp == mx) & (resp > 0))
    order = np.argsort(-resp[yy, xx])[:top]
    return np.column_stack([xx[order], yy[order]])
