# Rounds 2-3: learned detector, pretrained vs scratch, local adaptation

Three tracks I wanted to compare: (1) a COCO-pretrained detector, (2) no outside weights at all, pushed as far as it goes, (3) human-assisted: label a small patch of a new photo and adapt. Plus the brightness-profile / gradient feature tests.
Code and training by ChatGPT from my plan; I hand-edited the brightness-profile / gradient tests. Training ran in ChatGPT's workspace.

What's here:
- `project/` shared code used everywhere: `barnacles/core.py` (frame straightening, mask -> points, matching/F1), `barnacles/modern.py` (tiling, YOLO tile prediction), `data/` (photos, masks, `frames.json`), `modern_analysis/train_detector.py` (trains the main model)
- `models/` one-photo models (`pretrained_from1/2`, `scratch_from1/2`), adapted models (`*_adapt_to1/2`), `scratch_mixed_styles` (scratch, both photos)
- `results/` everything these runs saved (`detector_summary.json`, `detector_hybrid.json`, `profile_transfer.json`, `mixed_scratch_summary.json`, `*_selection.json` thresholds)
- `run_profiles.py`, `tune_profiles.py`, `profile_features.py`, `show_profiles.py` brightness profiles + gradients
- `run_detectors.py` pretrained vs scratch + adaptation, `run_mixed_scratch.py` scratch on both photos
- `adapt_new_image.py` adapt a model with labelled patches of your own photo, `predict_pilot.py` run any model on a photo, `audit_count.py` random-tile audit, `verify_pilot.py` checks saved outputs

Results used in the main notebook (§4, §6, §7): one photo -> the other, pretrained 0.583 / 0.795 vs scratch 0.165 / 0.184; profiles 0.041 / 0.062 -> profiles + gradients 0.238 / 0.354; 12.5% patch adaptation 0.796 -> 0.905 (photo 1) and 0.573 -> 0.751 (photo 2).

Run (training needs PyTorch + Ultralytics):
```bash
pip install -r requirements.txt   # plus torch + ultralytics
python run_profiles.py
python run_detectors.py
python run_mixed_scratch.py
cd project && python modern_analysis/train_detector.py --mode mixed --fraction 0.5 --epochs 100   # the main model
```
Run any saved model on a photo: `python predict_pilot.py --image project/data/img2.png --model models/pretrained_from1.pt --threshold 0.25 --output output/photo2` (thresholds are in `results/*_selection.json`).
