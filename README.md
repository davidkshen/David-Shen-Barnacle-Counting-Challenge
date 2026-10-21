# Barnacle counter (DALI Data Challenge 26F)

**David Shen**

NPS scientists count barnacles inside a fixed green frame by hand, often 1,000+ per photo. This repo is a counting assistant:
1. You mark the frame's 4 inner corners.
2. A detector puts a dot on every barnacle.
3. A person can then fix mistakes, or hand-count 16 random tiles (about a quarter of the frame) to get the count with an error bar.

Step 3 is tested in simulation; it isn't wired into the script yet.

**Bottom line:** worth pursuing as an assistant that does most of the counting, not as a replacement for a person.

The full story (what I tried, what failed, why I changed course) is in **[`barnacle_counting.ipynb`](barnacle_counting.ipynb)**. This README is the short version.

## Results

All numbers are for the main model (`models/mixed_0.5_yolo11n.pt`, threshold 0.37) unless the row says otherwise. Scoring: a dot is correct if it lands on a barnacle. F1 is 1.0 when every barnacle is found and there are no extra dots.

| Test | What it tells you | Result | Source |
|---|---|---|---|
| Held-out right strips (x ≥ 832) of the 2 labelled photos | Did it learn these photos? | **F1 0.941**: 408 dots for 406 barnacles; 383 correct, 25 extra, 23 missed | `results/check_main_model.json` |
| `unseen_img1`, a photo it never saw, run through the normal pipeline and scored on the ~10% I counted by hand (135 barnacles) | Does it work on a new photo? | **130 dots: 128 correct, 2 extra, 7 missed, F1 0.966.** Under the stricter rule (dot within ~5 px of the mark's centre): 0.928 | `results/marked_area_mixed_0.5_yolo11n.json` |
| `unseen_img1`, whole frame | How many does it count? | 1,143 (only the marked area was checked) | `results/check_main_model.json` |
| 97 close-up photos from 9 other survey groups (Roboflow, whole-shell labels), every photo shrunk ×0.7 | Very different photos? | **F1 0.757** (95% range 0.598–0.831). Per-photo count error 24%; 14% too few overall | `results/recorded/main_external.json` |
| Same type of model trained on 1 photo, tested on the whole other photo | Does pretraining matter? | Pretrained: 0.583 (photo 1→2) and 0.795 (2→1). From scratch: 0.165 and 0.184 | `results/recorded/detector_summary.json` |
| Random audit: hand-count 16 of 64 tiles. Simulated with the photo-2-only model on photo 1 | How close does the corrected total get? | ±9.0%; the interval contained the true count 94% of the time. The same tiles counted with no model: ±12.8% | notebook §10, `results/recorded/audit_sim.json` |

**Read these carefully:**
- **0.941 is not a new-photo score.** The model trained on other parts of the same two photos. It uses the original strip-only tiling; tiling the whole frame, the way `predict.py` does, gives 0.950 on the same strips.
- **The new-photo result is one area of one photo, counted by one person (me).**
  - **The scoring rule was picked after looking.** The "same barnacle" rule was chosen after seeing results, so the stricter rule set beforehand is always shown next to it.
  - **I counted the area twice.** My first count (138 marks, which left out every barnacle cut by the edge) gave 0.948, and 0.754 under the strict rule. I then re-marked it with the centre-inside rule (135 marks, placed closer to each opening's centre) after seeing those results. The first version's results are kept in `experiments/Barnacle_Brief_Ideas_Study/results/v1_marks_138/`.
- **The audit numbers are simulations.** Mask labels stood in for a person, and nobody was timed. In ChatGPT's test on the outside photos, the "95%" intervals only held the true count 80% of the time with 8 tiles and 89% with 16 (`results/recorded/chatgpt_round5_reported.json`). Treat the interval as a guide, not a guarantee.

## What the final pipeline actually runs

`scripts/predict.py` → `barnacle_counter`:
1. **Straighten the frame** from the 4 corners into a 1024×1024 square (`frames.rectify`). The frame is a fixed size, so the same real-world size always maps to the same number of pixels.
2. **Cut overlapping tiles:** 256 px, overlapping by 64 px (`tiles.windows`).
3. **Run YOLO11n** on each tile, enlarged to 512 px (`detector.Detector`).
   - The model is COCO-pretrained, then fine-tuned for 100 epochs (settings in `models/args.yaml`).
   - It was trained on a random half of the tiles from the left strip (x < 384) of both labelled photos. The split is done by `train_detector.py --fraction 0.5`.
4. **Keep each dot only in the tile it belongs to,** then merge duplicates (`tiles.tile_predict`).
5. **Keep dots above 0.37.** That threshold was picked on the validation strip, never the test strip (`models/default.json`).

**Not in the final pipeline:**
- **My original ring/slit/brightness filters.** No hand-made filter runs before or after YOLO. Convolution survives only as YOLO's own learned filters.
- **Hough circles, the hand-feature classifiers and the label-free tweaks.** Tested, then dropped.
- **Patch fine-tuning (adaptation), threshold re-picking and the audit.** These are separate steps, tested in experiments, not part of `predict.py`.

## Files

```
barnacle_counting.ipynb   main notebook: the reasoning + evidence, runs on a CPU in under a minute
barnacle_counter/         the code the notebook and scripts import
  frames.py   straighten the frame        labels.py   mask outlines -> one point per barnacle
  tiles.py    tiling + merging dots        detector.py load a model, count a photo
  metrics.py  matching, F1, count error    audit.py    random-tile audit estimate
  marks.py    read my hand marks, find the crop inside unseen_img1
  plots.py    overlays/charts              baselines.py my first ring-filter idea (demo only)
  yolo_np.py, ptload.py   YOLO11 in plain NumPy, so nothing needs PyTorch
scripts/
  predict.py            count one photo -> dots.csv, overlay.png, count.json
  check_main_model.py   re-checks 0.941 and 1,143
  score_marked_area.py  scores a model on my hand-counted area of unseen_img1
models/       mixed_0.5_yolo11n.pt (main), default.json (threshold), args.yaml + results.csv (training run),
              photo2_only_pretrained.pt (+ .json) = a model that never saw photo 1, used for the audit demo
data/         img1/img2 + masks, frames.json (frame corners), unseen_img1/2,
              unseen_img2_marked_135.png (my final hand count), unseen_img2_marked_138_first_try.png (first count)
results/      outputs of the scripts + recorded/ (saved results from the experiments, see results/README.md)
experiments/  every earlier round (see the table at the bottom)
```

The `model` path inside `models/default.json` is from the original training folder. The scripts load `models/mixed_0.5_yolo11n.pt` directly.

## How to run

```bash
pip install -r requirements.txt
```

**1. Just look at the results.** Open `barnacle_counting.ipynb` on GitHub; the outputs are saved in it. To re-run it, `jupyter notebook barnacle_counting.ipynb`, then Restart & Run All. It needs no GPU and no PyTorch. Cells marked **[runs]** compute live; cells marked **[saved]** load `results/recorded/`.

**2. Run the model.** CPU only, about 6 s per photo:
```bash
python scripts/predict.py --image data/unseen_img1.png --out output/unseen_img1
#   -> output/unseen_img1/dots.csv, overlay.png, count.json  (1,143 dots)
python scripts/predict.py --image my_photo.png --corners "[[TLx,TLy],[TRx,TRy],[BRx,BRy],[BLx,BLy]]"
python scripts/check_main_model.py      # expect F1 0.941 on the strips and 1,143 on unseen_img1
python scripts/score_marked_area.py     # expect 130 dots, F1 0.966 (strict 0.928)
```
- **You get a count, not an accuracy.** An F1 needs hand labels.
- **`--cropped` assumes the image is the whole frame.** It treats the photo as already being exactly the inside of the frame; a zoomed-in piece will have barnacles that are too big.
- **`--backend ultralytics`** uses the real library instead of the NumPy copy. It gives the same dots and needs PyTorch.

**3. Retrain or re-run the experiments.** Training needs PyTorch + Ultralytics:
```bash
cd experiments/Barnacle_Three_Track_Study/project
python modern_analysis/train_detector.py --mode mixed --fraction 0.5 --epochs 100
```
- **Training output.** Weights go to `project/modern_analysis/runs/` and the chosen threshold to `project/modern_analysis/results/`.
- **Each `experiments/` folder has a short README with its run order.**
- **The outside-photo experiments need the Roboflow export.** It's "Barnacle species identification2" (Reef Analysis Trial, Roboflow Universe, CC BY 4.0). It isn't included; see `experiments/Barnacle_Brief_Ideas_Study/README.md`.

## How I got here (details in the notebook)

1. **Describe it by hand** (notebook §3–4).
   - My first idea: a barnacle is a bright ring with a dark slit, so look for that with filters. It scored near zero: 0.000 / 0.008 for the strict rule, 0.09 / 0.14 for filters + a classifier.
   - Real barnacles break the rule: uneven lighting, moss, crowding, and empty shells that have perfect rings but don't count.
   - Brightness profiles + gradients helped (0.04 / 0.06 → 0.24 / 0.35). But the step that picks candidate spots was already missing barnacles.
2. **Let a model learn it** (§5).
   - YOLO11n, COCO-pretrained, with tiling: 0.941 on the held-out strips.
   - Paul (DALI data dev head) said pretrained models are fine if I explain why I picked mine. Outside data is fine if it actually helps for the barnacles I'm counting.
3. **Question the 0.941** (§6).
   - The model had already been exposed to both photos. Trained on one and tested on the whole other photo, it drops to 0.58 / 0.80 (from scratch: 0.16 / 0.18).
   - Two photos means two sets of conditions. That's too few to know how well anything generalizes.
4. **Local adaptation** (§7).
   - The reasoning: training on pieces of both photos worked, and training on one didn't, so exposure to a photo's own conditions matters. Nobody will label a whole photo they want automated, so label a small patch of it and fine-tune.
   - Fine-tuning on a 12.5% patch, scored on that photo's test strip: 0.80 → 0.90 (photo 1) and 0.57 → 0.75 (photo 2).
   - Labelling 1–2 tiny squares and only re-picking the threshold helped photo 1 and made photo 2 worse.
5. **Outside photos** (§8).
   - The main model holds up (0.757) once the photo is shrunk to the right size.
   - A ChatGPT run trained a model on the outside photos plus the left strips of my two photos: 0.804 on its own split of the outside photos. That model dropped to about 0.70 / 0.68 on my photos' test strips, run over each full photo rather than strip-only. Different split and protocol; numbers in `results/recorded/chatgpt_round5_reported.json`, not re-run.
6. **A real new-photo check** (§9). I hand-counted part of `unseen_img1`: 0.966 (strict rule: 0.928).
7. **A count with an error bar** (§10). A random tile audit turns a model count into a count ± interval. The interval still needs checking with real counts.

## Done vs not done

**Done:**
- The main model and its re-check.
- One-photo transfer tests.
- Adaptation, with simulated labels.
- The challenge's idea list: OpenCV, sklearn, CNN from scratch vs pretrained.
- Outside-photo tests.
- My hand count on part of `unseen_img1`.
- Audit simulations.

**Not done / proposed:**
- A real person reviewing or auditing, with timing.
- A random-tile hand count of `unseen_img1`, which would check the 1,143 without bias.
- More hand-counted new photos.
- Telling live barnacles from empty shells.
- Hooking the review page up to the main model.
- A proper "more outside data → better" curve. This needs new training runs.
- Mask R-CNN / Segment Anything. Considered, not run: they draw outlines, and counting only needs points.

## Limits

- **Two labelled photos,** both used to train the main model.
- **The new-photo check is one area of one photo.** That area is what the `unseen_img2` crop covers, not a random sample.
- **Outside photos are a different target and a rough split.** They're labelled as whole shells, not openings, and grouped by survey using file names, so those scores may be a bit optimistic.
- **The model doesn't know live from dead.** A dark opening doesn't mean dead. Empty shells aren't outlined in the masks, so they don't count.

## How this was built (AI use)

- **I brainstormed the process.** That covers the overall plan and each change of direction:
  - ring/slit filters → brightness gradients → a learned detector;
  - questioning the 0.941 → local adaptation → outside data and the audit.
- **AI tools wrote most of the code from my direction.** I reviewed what came back and refined it, partly by directing the changes and partly by hand. I hand-edited the code for the gradient tests and the other visual-feature tests (brightness profiles, ring/slit filters) that I ran to see if more detail from the image would help. Changes I directed:
  - loosening the ring parameters;
  - adding gradients;
  - switching to a pretrained detector;
  - testing on a photo the model hadn't seen;
  - the three-track plan (pretrained / no outside weights / human-assisted).
- **The AI ran the rest of the code and all the model training in its own workspace.**
  - ChatGPT (OpenAI) did rounds 1–3, the first outside-photo evaluation and the broader-data run. That includes training the main model.
  - Claude (Anthropic) did rounds 4 and 6, the re-checks, and this repo (package, notebook, README).
- **Done by me directly:**
  - hand-editing the gradient / visual-feature test code;
  - checking the rules with Paul;
  - finding and exporting the outside dataset;
  - both hand counts on `unseen_img2`.

## experiments/

The earlier rounds: code, saved results and a short README each. Old reports were removed; everything they showed is in the notebook and this README. The code itself isn't re-annotated, so its style differs from `barnacle_counter/`.

| Folder | Round | Made by | What |
|---|---|---|---|
| `Barnacle_Assistant/` | 1 | ChatGPT; I hand-edited the filter/gradient tests | Hand-made ring/slit/gradient filters + tree classifier, review page, audit code |
| `Barnacle_Three_Track_Study/` | 2–3 | ChatGPT; I hand-edited the brightness-profile / gradient tests | Shared code (`project/barnacles/`), training script, one-photo / scratch / adapted models, profile/gradient tests |
| `Barnacle_Improvement_Study/` | 4 | Claude | NumPy YOLO, diagnosis, small-label + audit simulations, timed review page |
| `Barnacle_Brief_Ideas_Study/` | 6 | Claude | Challenge idea list, Roboflow benchmark, main-model and new-photo tests |

Two earlier ChatGPT rounds had no code worth keeping, so only their numbers are kept: the strict ring + slit rule tests (round 1b, `results/recorded/ring_rule_tests.json`; I hand-edited those feature tests too) and the broader-data training run (round 5, `results/recorded/chatgpt_round5_reported.json`, not re-run because its weights were never available).

## Outside data (Roboflow)

**What:** "Barnacle species identification2" by Reef Analysis Trial on Roboflow Universe (link: ___). 561 close-up photos of barnacles on Welsh rocky shores, with 74,805 labelled boxes. Licence CC BY 4.0; images © Natural Resources Wales, OGL v3. I found it and exported it in YOLO format.

**Why this one:** Paul said outside data is fine as long as it actually helps for the barnacles you're counting. This set is acorn barnacles on intertidal rock, like the NPS photos, and one of its classes is *Semibalanus balanoides*, a North Atlantic barnacle that also lives on the US east coast.

**What I checked before using it:**
- The export only had class numbers. I worked out the 9 class names from Roboflow's alphabetical order and the box sizes, then kept the 5 adult barnacle classes and dropped cyprids, spat and snails.
- The labels are boxes around whole shells, not the openings my masks outline, and the photos are much closer up. So each photo gets shrunk by one factor (×0.7, picked on tuning photos) before the model sees it.
- Roboflow's own train/test split mixes photos from the same survey. I re-split by survey (same date, or same site and year, from the file names): 438 train / 26 tuning / 97 test photos (31 / 5 / 9 survey groups). It's a rough grouping, so the test scores may be a bit optimistic.

**How it was used:** to test the main model on very different photos (F1 0.757 on the 97 test photos, never trained on them), and to train one comparison classifier (0.751). An earlier ChatGPT run also trained a detector on it. That model did better on these photos (0.804 on its own split) but worse on my two photos (~0.70 / 0.68), so outside data helped there and hurt here.

**Not included in this repo.** Download it from Roboflow Universe (YOLO export) and unzip it into `experiments/roboflow_export/` to re-run the outside-photo tests.

Outside photos: Reef Analysis Trial, "Barnacle species identification2" (Roboflow Universe), CC BY 4.0; images © Natural Resources Wales, OGL v3.