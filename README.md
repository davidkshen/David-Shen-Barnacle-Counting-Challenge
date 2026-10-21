# Barnacle counter (DALI Data Challenge 26F)

**David Shen**

NPS scientists count barnacles inside a fixed green frame by hand, often 1,000+ per photo. This repo is a counting assistant:
1. You mark the frame's 4 inner corners. This sets the area to count, so a scientist can also count just part of a photo.
2. A detector puts a dot on every barnacle.
3. A person fixes mistakes, or hand-counts 16 random tiles (about a quarter of the frame) to get the count with an error bar. Being able to correct the dots lets researchers reach the accuracy their work needs. (Step 3 is tested in simulation; it isn't wired into the script yet.)

**Bottom line:** worth pursuing as an assistant that does most of the counting, but not yet good enough to replace a person.

The full story (what I tried, what failed, why I changed course) is in **[`barnacle_counting.ipynb`](barnacle_counting.ipynb)**. This README is the short version.

## Results

Main model (`models/mixed_0.5_yolo11n.pt`, threshold 0.37) unless noted. A dot is correct if it lands on a barnacle. F1 = 1.0 means every barnacle found and no extra dots.

| Test | Result | Source |
|---|---|---|
| Held-out right strips (x ≥ 832) of the 2 labelled photos | **F1 0.941**: 408 dots for 406 barnacles (383 correct, 25 extra, 23 missed) | `results/check_main_model.json` |
| New photo (`unseen_img1`), scored on the ~10% I hand-counted (135 barnacles) | **F1 0.966**: 130 dots (128 correct, 2 extra, 7 missed). Strict rule (within ~5 px of the centre): 0.928 | `results/marked_area_mixed_0.5_yolo11n.json` |
| `unseen_img1`, whole frame | 1,143 barnacles (only the marked area was checked) | `results/check_main_model.json` |
| 97 outside photos from 9 other surveys (Roboflow), shrunk ×0.7 | **F1 0.757** (95% range 0.60–0.83). Count off 24% per photo, 14% low overall | `results/recorded/main_external.json` |
| Trained on 1 photo, tested on the other | Pretrained 0.58 / 0.80. From scratch 0.16 / 0.18 | `results/recorded/detector_summary.json` |
| Simulated audit: hand-count 16 of 64 tiles (photo-2-only model on photo 1) | ±9.0%; the interval held the true count 94% of the time. No model: ±12.8% | `results/recorded/audit_sim.json` |

**Important notes:**
- **0.941 isn't a new-photo score.** The model trained on parts of the same two photos, then was tested on the parts it didn't train on. Tiling the whole frame (the way `predict.py` does) gives 0.950 on the same strips.
- **The new-photo check is one area of one photo, counted by me.** The "same barnacle" scoring rule was picked after seeing results, so the strict rule (set beforehand) is always shown next to it. I counted the area twice: my first count (138 marks) gave 0.948 (strict 0.754). After seeing that, I re-marked it with the centre-inside rule (135 marks). First-count results: `experiments/Barnacle_Brief_Ideas_Study/results/v1_marks_138/`.
- **The audit numbers are simulations.** Mask labels stood in for a person, and nobody was timed. In ChatGPT's test on the outside photos, the "95%" interval only held the true count 80% of the time with 8 tiles and 89% with 16. It's a guide, not a guarantee.

## What is actually happening

`scripts/predict.py` → `barnacle_counter`:
1. **Straighten the frame** from the 4 corners into a 1024×1024 square, so a barnacle is always about the same size in pixels.
2. **Cut overlapping tiles** (256 px, 64 px overlap) and enlarge each to 512 px.
3. **Run YOLO11n** on each tile. It's COCO-pretrained, then fine-tuned for 100 epochs on a random half of the tiles from the left strip (x < 384) of both labelled photos (`models/args.yaml`).
4. **Merge:** keep each dot only in the tile that owns it, then remove duplicates.
5. **Keep dots above 0.37.** That threshold was picked on the validation strip, never the test strip.

**Not in the final pipeline:** my hand-made ring/slit/brightness filters, Hough circles, the hand-feature classifiers, and the Roboflow photos (used only as a test). Patch fine-tuning, threshold re-picking and the audit were tested in experiments but aren't part of `predict.py`.

**Why COCO-pretrained YOLO11n**
- **Pretraining clearly helps.** Trained on one photo and tested on the other: 0.58 / 0.80 pretrained vs 0.16 / 0.18 from scratch. Trained on both photos: 0.941 vs 0.889.
- **It's a detector.** One box per barnacle gives a centre to count and check.
- **Nano is the smallest YOLO11.** It fits the small training set and runs on a CPU in about 6 s per photo, even without PyTorch.
- **Not tested:** bigger YOLO sizes or other detectors. It worked; I can't say it's the best.

**Outside data:** [Barnacle species identification2](https://universe.roboflow.com/reef-analysis-trial/barnacle-species-identification2) (Reef Analysis Trial, Roboflow Universe, CC BY 4.0; images © Natural Resources Wales, OGL v3). It has 561 close-up photos of barnacles on Welsh shores, with 74,805 labelled boxes. It was the closest public labelled barnacle set I found, and I found and exported it myself.
- **Training on it didn't help for my barnacles.** A ChatGPT run trained on it plus my photos' left strips. It got 0.804 on the outside photos but only ~0.70 / 0.68 on my test strips, where the main model gets 0.950 scored the same way. So the final model doesn't train on it.
- **Used as a test instead:** 0.757 on 97 photos from 9 surveys the model never saw.
- **Prep:** the export only had class numbers, so I worked out the names and kept the 5 adult barnacle classes. Roboflow's split mixes photos from the same survey, so I re-split by survey (438 / 26 / 97 photos). The photos are close-ups, so each was shrunk ×0.7 (picked on the tuning photos).
- **Why it likely didn't help:** its labels are whole shells, not the openings my masks outline; its photos are close-ups, not a wide frame; and it's a different shore.
- **Not included in this repo.** To re-run, download the YOLO export and unzip it into `experiments/roboflow_export/`.

## Files

```
barnacle_counting.ipynb   the notebook: reasoning + evidence, runs on a CPU in under a minute
barnacle_counter/         the code: frame, tiles, detector, metrics, audit, hand marks, plots
                          (yolo_np.py + ptload.py run YOLO11 in plain NumPy, so no PyTorch)
scripts/                  predict.py (count a photo), check_main_model.py, score_marked_area.py
models/                   mixed_0.5_yolo11n.pt (main) + default.json (threshold) + training settings;
                          photo2_only_pretrained.pt (never saw photo 1, used for the audit demo)
data/                     labelled photos + masks, frame corners, unseen photos, my hand counts (135 final, 138 first try)
results/                  script outputs + recorded/ (saved experiment results, see results/README.md)
experiments/              earlier rounds (table at the bottom)
```

## How to run

```bash
pip install -r requirements.txt
```

**1. Just look.** Open `barnacle_counting.ipynb` on GitHub; the outputs are saved. To re-run, Restart & Run All (no GPU, no PyTorch). **[runs]** cells compute live; **[saved]** cells load `results/recorded/`.

**2. Run the model** (CPU, about 6 s per photo):
```bash
python scripts/predict.py --image data/unseen_img1.png --out output/unseen_img1   # 1,143 dots
python scripts/predict.py --image my_photo.png --corners "[[TLx,TLy],[TRx,TRy],[BRx,BRy],[BLx,BLy]]"
python scripts/check_main_model.py     # expect 0.941 and 1,143
python scripts/score_marked_area.py    # expect 130 dots, F1 0.966 (strict 0.928)
```
You get a count, not an accuracy (an F1 needs hand labels). `--cropped` means the image is already exactly the inside of the frame. `--backend ultralytics` uses the real library: same dots, needs PyTorch.

**3. Retrain** (needs PyTorch + Ultralytics):
```bash
cd experiments/Barnacle_Three_Track_Study/project
python modern_analysis/train_detector.py --mode mixed --fraction 0.5 --epochs 100
```
Each `experiments/` folder has a short README with its run order.

## How I got here

1. **Hand rules** (notebook §3–4). A barnacle = bright ring + dark slit, found with filters. It scored near zero (0.000 / 0.008; filters + classifier 0.09 / 0.14), because lighting, moss, crowding and empty shells break the rule. Gradients helped (0.04 / 0.06 → 0.24 / 0.35), but the candidate-finding step was already missing barnacles.
2. **Let a model learn it** (§5). YOLO11n with tiling: 0.941.
3. **Question the 0.941** (§6). Trained on one photo and tested on the other, it drops to 0.58 / 0.80. Two photos are too few to know how well it generalizes.
4. **Local adaptation** (§7). If seeing a photo's own conditions matters, label a small patch and fine-tune. A 12.5% patch: 0.80 → 0.90 (photo 1), 0.57 → 0.75 (photo 2). Only re-picking the threshold from 1–2 tiny squares helped photo 1 but hurt photo 2.
5. **Outside photos** (§8). The main model holds at 0.757; training on them hurt my photos.
6. **Real new-photo check** (§9). 0.966 (strict 0.928) on my hand-counted area.
7. **Count with an error bar** (§10). A random-tile audit gives count ± interval. The interval still needs checking with real counts.

## Done / not done / limits

**Done:** main model + re-check, one-photo transfer, adaptation (simulated labels), the challenge's idea list (OpenCV, sklearn, scratch vs pretrained CNN), outside-photo tests, my hand count on part of `unseen_img1`, audit simulations.

**Not done:**
- A real person reviewing or auditing, timed.
- A random-tile hand count of `unseen_img1` to check the 1,143 without bias. I didn't have time: it would take around 30 minutes, with mistakes on my end too.
- More hand-counted new photos.
- Telling live barnacles from empty shells. I don't have the biology background to tell dead or empty ones apart reliably, so I couldn't build a rule for it.
- Hooking the review page up to the main model.
- A proper "more outside data → better" curve (needs new training runs).
- Mask R-CNN / Segment Anything: considered, not run. They draw outlines, and counting only needs points.

**Limits:** only two labelled photos, both used in training. The new-photo check is one non-random area of one photo. The outside photos use a different target (whole shells) and a rough survey split, so those scores may be a bit optimistic. The model can't tell live from dead; empty shells aren't in the masks, so they don't count.

## How this was built (AI use)

- **I brainstormed the process** and each change of direction: ring/slit filters → gradients → a learned detector; questioning the 0.941 → local adaptation → outside data and the audit.
- **Done by me directly:** hand-editing the gradient / visual-feature test code (brightness profiles, ring/slit filters), checking the rules with Paul, finding and exporting the outside dataset, and both hand counts on `unseen_img2`.
- **AI tools wrote a good chunk of the code from my direction.** I reviewed and refined it, partly by directing changes and partly by hand. Changes I directed: loosening the ring parameters, adding gradients, switching to a pretrained detector, testing on a photo the model hadn't seen, and the three-track plan (pretrained / no outside weights / human-assisted).
- **The AI ran the rest of the code and all model training in its own workspace.** ChatGPT (OpenAI): rounds 1–3, the first outside-photo test and the broader-data run, including training the main model. Claude (Anthropic): rounds 4 and 6, the re-checks, and this repo (package, notebook, README).

## experiments

Earlier rounds: code, saved results and a short README each. Their code style differs from `barnacle_counter/`.

| Folder | Round | Made by | What |
|---|---|---|---|
| `Barnacle_Assistant/` | 1 | ChatGPT; I hand-edited the filter/gradient tests | Ring/slit/gradient filters + tree classifier, review page, audit code |
| `Barnacle_Three_Track_Study/` | 2–3 | ChatGPT; I hand-edited the brightness-profile/gradient tests | Shared code, training script, one-photo / scratch / adapted models |
| `Barnacle_Improvement_Study/` | 4 | Claude | NumPy YOLO, small-label + audit simulations, timed review page |
| `Barnacle_Brief_Ideas_Study/` | 6 | Claude | Challenge idea list, Roboflow benchmark, new-photo tests |

Two ChatGPT rounds had no code worth keeping, so only their numbers are saved: the strict ring + slit rule (round 1b, `results/recorded/ring_rule_tests.json`; I hand-edited those tests too) and the broader-data training run (round 5, `results/recorded/chatgpt_round5_reported.json`, not re-run because its weights were never available).
