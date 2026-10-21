# results/

Top level = made by the scripts in this repo (`scripts/`). Re-running them overwrites these.

| file | made by | what |
|---|---|---|
| `check_main_model.json` | `scripts/check_main_model.py` | main model on the held-out strips (0.941) + count on unseen_img1 (1,143) |
| `marked_area_mixed_0.5_yolo11n.json` | `scripts/score_marked_area.py` | main model on my hand-counted area of unseen_img1 (0.966, strict 0.928) |
| `marked_area_photo2_only_pretrained.json` | same, `--model models/photo2_only_pretrained.pt --threshold 0.33` | the photo-2-only model on the same area (0.912) |

`recorded/` = copies of results saved by the earlier experiments (code in `experiments/`). The notebook loads these instead of re-running hours of work.

| file | from | what |
|---|---|---|
| `round1_filter_metrics.json` | `Barnacle_Assistant/results/metrics.json` | hand-made ring/slit/gradient filters + tree classifier, train on one photo, test on the other |
| `profile_transfer.json` | `Barnacle_Three_Track_Study/results/` | brightness profiles vs profiles + gradients, one photo -> the other |
| `detector_summary.json` | same | YOLO11n pretrained vs scratch trained on one photo; `evaluations[0]` = whole other photo, `[1]` = other photo's test strip |
| `mixed_scratch_summary.json` | same | scratch YOLO trained on both photos, test strips (0.889) |
| `detector_hybrid.json` | same | fine-tuning on a 12.5% patch of the new photo (adaptation) |
| `adapt_budget.json` | `Barnacle_Improvement_Study/results/` | label 1-8 small squares, re-pick threshold only (20 random draws each) |
| `audit_sim.json`, `combined_workflow_sim.json` | same | random-tile audit simulations (mask labels stand in for a person) |
| `numpy_validation.json` | same | NumPy YOLO vs saved Ultralytics outputs (14/14 match) |
| `classical_cv.json`, `probe_dali.json`, `creative_dali.json`, `data_exploration.json` | `Barnacle_Brief_Ideas_Study/results/` | the challenge's idea list (OpenCV, sklearn), label-free tweaks, mask stats |
| `detectors_external.json`, `probe.json`, `main_external.json` | same | outside (Roboflow) photos: one-photo models, classifier trained on Roboflow, main model |
| `unseen2_in_pipeline.json` | same (`unseen2_in_pipeline.py`) | every saved model on my hand-counted area of unseen_img1, normal pipeline |
| `main_unseen2.json`, `unseen2_same_barnacle.json` | same | the same area run as a separate cropped image |
| `ring_rule_tests.json` | ChatGPT's ring-rule tests (round 1b) | strict ring + slit rule, scored on each photo itself (0.000 / 0.008) |
| `chatgpt_round5_reported.json` | ChatGPT's round-5 report, numbers copied, not re-run | broader-data training, its own Roboflow split, its audit coverage (80% / 89%) |
