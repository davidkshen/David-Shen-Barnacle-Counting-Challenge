# Round 1: describe barnacles by hand

My first try: barnacles look like a bright ring with a dark slit, so build hand-made filters for that (ring, slit, gradient), feed the scores to a tree classifier, and add a review page and a random-tile audit.
Code written by ChatGPT from my ideas; I hand-edited the filter / gradient test code.

Result: it didn't transfer. Trained on one photo, tested on the other: F1 0.091 (photo 1 -> 2) and 0.140 (2 -> 1), in `results/metrics.json` (copied to `../../results/recorded/round1_filter_metrics.json`, notebook §3).

What's here:
- `barnacles/` the filters (`core.py`), classifier, audit and review-page code
- `run_experiment.py` trains on one photo, tests on the other, writes `results/`
- `predict.py` runs a saved classifier on a photo and makes a review page
- `compare_importance.py` compares model importance maps with human-marked maps (idea from a Serre talk; no human maps were collected)
- `data/` the challenge photos + `unseen_img1/2`, `results/` saved outputs, `tests/` unit tests

Run (CPU, no PyTorch):
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py
python predict.py --image data/unseen_img1.png --model results/model_from_img1.joblib --output review_exports/unseen
```
