# Round 4: running without PyTorch, small-label and audit simulations

Code by Claude.
- `npyolo/` YOLO11 in plain NumPy (checked against 14 saved Ultralytics outputs, `results/numpy_validation.json`). The clean copy the main repo uses is in `../../barnacle_counter/yolo_np.py`.
- `experiments/` diagnosis of why one photo -> the other fails, threshold re-picking with 1-8 labelled squares (`adapt_budget.py`), random-tile audit simulations (`audit_sim.py`, `combined_workflow_sim.py`), review-effort simulation, `main_model_eval.py` (re-checks 0.941 and 1,143)
- `review/` a timed review / blind-audit page (`make_review_kit.py`), built and tested with other saved models, not hooked up to the main model
- `local_training/run_local.py` training experiments that need PyTorch (seed spread, matched scratch run, small-budget fine-tuning). Not run.
- `external/inspect_yolo_export.py` checks a Roboflow export before using it

All simulations use the mask labels in place of a person. Nobody was timed.

Run (CPU, no PyTorch):
```bash
bash run_all.sh                                   # rebuilds results/ (needs ../Barnacle_Three_Track_Study)
python experiments/main_model_eval.py             # expect 0.941 and 1,143 (uses ../../models/)
```
Predictions get cached in `cache/` (ignored by git).
