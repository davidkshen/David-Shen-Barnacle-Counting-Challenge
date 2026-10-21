"""The label-free Otsu threshold (declared in creative_dali.py for DALI) applied unchanged to the
external test photos, per photo, at each model's chosen fixed scale. No labels, no new inference."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import detectors_external as DE  # noqa: E402
import creative_dali as CD  # noqa: E402
OUT = Path(__file__).resolve().parent / 'results'


def main():
    de = json.loads((OUT / 'detectors_external.json').read_text())
    test = C2.rf_rows('test')
    refs = [C2.rf_targets(r) for r in test]
    groups = [r['group'] for r in test]
    out = {}
    for tag in ('pretrained_from2', 'pretrained_from1', 'scratch_mixed_styles'):
        f = de[tag]['fixed_scale']['factor']
        P = [DE.predict_at(tag, r, f) for r in test]
        kept = []
        for p in P:
            t = CD.otsu_thr(p) if (p[:, 3] > 0.01).sum() >= 10 else de[tag]['dali_threshold']
            kept.append(p[p[:, 3] >= t])
        m = dict(**C2.pooled(kept, refs), **C2.count_errors(kept, refs), f1_group_ci95=C2.group_bootstrap(kept, refs, groups))
        out[tag] = dict(factor=f, otsu_per_photo=m, dali_threshold_f1=de[tag]['fixed_scale']['dali_threshold']['f1'],
                        calibrated_f1=de[tag]['fixed_scale']['calibrated']['f1'])
        print(tag, f"Otsu per photo F1 {m['f1']:.3f} {m['f1_group_ci95']} (P {m['precision']:.2f} R {m['recall']:.2f}, count {m['weighted_abs_pct']:.0f}%) "
                   f"vs DALI threshold {out[tag]['dali_threshold_f1']:.3f}, calibrated {out[tag]['calibrated_f1']:.3f}", flush=True)
    C2.save(OUT / 'external_otsu.json', out)


if __name__ == '__main__':
    main()
