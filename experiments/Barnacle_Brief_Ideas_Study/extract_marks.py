"""Turn the user's red-marked copy of unseen_img2 into point labels (red_marks.json).
A mark = a connected group (8-neighbour) of pixels where the marked copy differs from the clean photo by
more than 60 (sum of |dR|+|dG|+|dB|). Centre = mean pixel position; L/W from the minimum-area rectangle.
Usage: python extract_marks.py MARKED.png CLEAN.png OUT.json"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi


def main(marked, clean, out):
    a = np.asarray(Image.open(marked).convert('RGB')).astype(int)
    b = np.asarray(Image.open(clean).convert('RGB')).astype(int)
    lab, n = ndi.label(np.abs(a - b).sum(-1) > 60, np.ones((3, 3)))
    marks = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        ys, xs = np.nonzero(lab[sl] == i)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) >= 5:
            (_, _), (w, h), _ = cv2.minAreaRect(pts)
            L, W = max(w, h), min(w, h)
        else:
            L = W = 1
        marks.append(dict(id=i, x=float(xs.mean()), y=float(ys.mean()), n=int(len(xs)), L=float(L), W=float(W)))
    json.dump(marks, open(out, 'w'))
    print(len(marks), 'marks')


if __name__ == '__main__':
    main(*sys.argv[1:4])
