# Round 6: the challenge's idea list, outside photos, the new-photo check

Code by Claude.

Two parts:
1. **The challenge's example ideas, all tested the same way** (train on one photo, test on the other): OpenCV kernels, Hough circles, ellipses, scikit-learn on frozen CNN features, plus a few label-free tweaks. Also an outside-photo benchmark built from the Roboflow export, split by survey.
2. **The main model on new photos:** the Roboflow test photos, and the part of `unseen_img1` I counted by hand (`unseen_img2` is a 1.6x crop of it; my marks are in `../../data/`).

Scripts (run from this folder, CPU only, no PyTorch; they find the other `Barnacle_*` folders and `../../models/` by path):

| script | what | time |
|---|---|---|
| `rf_bench.py` | builds the outside-photo benchmark (needs the Roboflow export in `../roboflow_export/` or `BARNACLE_RF_EXPORT`) | ~5 min |
| `explore_data.py` | size / spacing / crowding stats from the masks | ~1 min |
| `classical_cv.py` | LoG blobs, ring kernel, Hough circles, contour ellipses | ~25 min |
| `probe.py dali` / `probe.py rf` | frozen COCO features + scikit-learn classifier, trained on DALI or Roboflow | ~10 / ~40 min |
| `creative_dali.py`, `scale_estimators.py`, `external_otsu.py`, `external_extension.py`, `reliability_flags.py` | label-free tweaks (most didn't help) | minutes |
| `detectors_external.py`, `main_external.py` | saved DALI models and the main model on the outside photos | ~40 / ~60 min |
| `extract_marks.py MARKED CLEAN OUT` | my red marks -> point labels | seconds |
| `unseen2_in_pipeline.py` | every saved model on my hand-counted area, normal pipeline | ~5 min |
| `unseen2_eval.py`, `main_unseen2.py`, `unseen2_same_barnacle.py`, `probe_unseen2.py`, `probe_apply.py` | same area run as a separate cropped image | ~30 min |
| `overlay.py`, `overlay_pipeline.py` | overlay pictures | seconds |

`results/` has everything these wrote. `results/v1_marks_138/` = the same scoring with my first (138-mark) count. The main notebook loads copies from `../../results/recorded/`.

Outside photos: Reef Analysis Trial, "Barnacle species identification2" (Roboflow Universe), CC BY 4.0; images © Natural Resources Wales, OGL v3. Not included.
