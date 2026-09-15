# Results

Generated 2026-09-15 13:19 by `scripts/make_results_md.py` directly from the run logs. Do not edit by hand - regenerate.

Commit: `8992ced`

## 1. Setup

### Source models

| checkpoint | architecture | used by | md5 |
|---|---|---|---|
| `panoptic_fpn_R50_cityscapes` | Panoptic FPN R50 (MTL: det + sem-seg) | all same-source experiments | `3a4df82ff7ab08b5` |
| `mask_rcnn_R50_cityscapes` | Mask R-CNN R50-FPN (detection specialist) | ST-D, specialist study only | `bee2df53a0757b8f` |
| `semantic_R50_cityscapes` | Semantic FPN R50 (segmentation specialist) | ST-S, specialist study only | `f09d1aa9d5acae21` |

All same-source comparisons use the **Panoptic FPN R50 MTL** checkpoint. Every baseline (TENT, CoTTA, AMROD) is re-run on that same checkpoint rather than quoted from its paper, so no comparison confounds the adaptation method with the source model.

### Protocols

| protocol | stream | evaluations |
|---|---|---|
| `cscLT` | (Fog -> Motion -> Snow -> Bright -> Defocus) x 10 | 50 |
| `acdcLT` | (Fog -> Night -> Rain -> Snow) x 10 | 40 |
| `csc12` | (Defocus -> Glass -> Motion -> Zoom -> Snow -> Frost -> Fog -> Bright -> Contrast -> Elastic -> Pixelate -> Jpeg) x 1 | 12 |
| `acdc4` | (Fog -> Night -> Rain -> Snow) x 1 | 4 |

No reset between domains or rounds; strictly online, batch size 1, one gradient step per image, each image seen once. Detection is scored with COCO mAP@0.5 on the Cityscapes 8 thing classes; segmentation with mIoU over 19 classes.

### Adaptation recipe (shared by all `Ours` rows)

```
mean teacher      EMA 0.9998, stochastic restore p=0.01
                  shared-trunk restore factor 0.1 (CTCMT_CROSS_TASK_FISHER)
pseudo-labels     AMROD dynamic per-class thresholds
                  init 0.80, min 0.70, alpha 1.3, gamma 0.95
                  ceiling 0.90 on ACDC, 0.80 on Cityscapes-C (E13a onward)
gate              AMROD score-EMA gate (score_em 0.5, gamma 0.7, thresh 1.4)
student view      strong augmentation; teacher/anchor keep the weak view
losses            det consistency 1.0 | seg soft-CE 1.0 | CT-CL 0.5 | CT-CR 0.3
                  CT-CR mode soft_seg_global, weight floor 0.2
                  class-balanced seg CE (beta 0.5), magnitude-preserving
                  anchor class-marginal KL 0.1
optimiser         SGD, lr 1e-3, momentum 0.9, weight decay 1e-4
```

`Iter.` is the estimated number of optimiser steps, from the fraction of sampled iterations that produced a loss. It is lower than the image count because the score-EMA gate skips images the teacher is already confident on.

## 2. Key findings

1. **On ACDC, multi-task adaptation is the best configuration on both tasks.** Full MTL leads detection-only by +0.6 mAP0.5 and +0.9 mIoU, and beats AMROD by ~+5 mAP0.5 and TENT/CoTTA by +7.8/+19.6 mIoU.

2. **On Cityscapes-C the sign flips: adding the segmentation loss costs detection.** Detection-only is +2.0 mAP0.5 and +4.7 mIoU *above* full MTL. The factorial below shows this is an interaction, not a main effect.

3. **The cause is not gradient conflict.** Mean cos(g_det, g_aux) on the shared trunk is *positive* (+0.18 to +0.21) and conflicts occur on under 10% of steps. Four arbitration mechanisms (S1 PCGrad-style projection, S2 CAGrad, S3 hard decoupling, S5 dynamic weighting) recover at most +0.9 mAP0.5 of the 2.0 gap; S5 is worse than doing nothing.

4. **Removing the cross-task losses does not explain it either.** C1 keeps only detection + plain segmentation soft-CE on the shared trunk and still loses ~80% of the gap.

5. **Routing the segmentation gradient off the shared trunk recovers the ceiling on Cityscapes-C** (S6: 26.7 vs detection-only 26.8, inside the ~0.5 noise floor) **but costs 2.2 mIoU on ACDC.** The intervention is benchmark-specific and is reported as a diagnostic, not as the method.

6. **Adaptation lives in the shared trunk.** Freezing the backbone and FPN (S4) costs 8.9 mAP0.5 and 3.8 mIoU, ruling out adapter-style parameter isolation.

## Table 1
**Cityscapes to Cityscapes-C, long-term &mdash; mAP0.5.** Five corruptions repeated ten times, no reset (50 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | R1 Fog | R1 Motion | R1 Snow | R1 Bright | R1 Defocus | R5 Fog | R5 Motion | R5 Snow | R5 Bright | R5 Defocus | R10 Fog | R10 Motion | R10 Snow | R10 Bright | R10 Defocus | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Source (no adaptation) | 33.0 | 9.5 | 0.2 | 16.4 | 6.6 | 33.0 | 9.5 | 0.2 | 16.4 | 6.6 | 33.0 | 9.5 | 0.2 | 16.4 | 6.6 | **13.2** | +0.0 | / |
| AMROD | 36.2 | 11.4 | 0.2 | 24.4 | 9.8 | 48.6 | 18.5 | 1.4 | 46.1 | 20.9 | 51.4 | 21.2 | 4.7 | 52.0 | 26.2 | **26.4** | +13.2 | / |
| Ours, full MTL (E11, thr 0.9) | 38.7 | 11.6 | 0.2 | 29.7 | 11.7 | 44.5 | 18.4 | 1.9 | 43.4 | 20.4 | 41.1 | 19.3 | 3.3 | 42.6 | 21.4 | **24.6** | +11.4 | 25.0k |
| Ours, full MTL (E13a, thr 0.8) | 38.5 | 12.2 | 0.2 | 29.4 | 11.1 | 45.4 | 19.2 | 3.1 | 43.3 | 19.7 | 41.8 | 19.9 | 6.0 | 41.6 | 20.0 | **24.8** | +11.6 | 25.0k |
| C1 &mdash; no cross-task losses | 36.9 | 11.3 | 0.2 | 24.3 | 10.1 | 48.1 | 18.3 | 1.1 | 44.3 | 19.6 | 48.8 | 20.7 | 1.6 | 48.0 | 23.3 | **25.2** | +12.0 | 25.0k |
| B1 &mdash; detection-only | 36.9 | 11.6 | 0.2 | 25.6 | 11.1 | 49.9 | 19.6 | 1.0 | 48.8 | 23.4 | 49.3 | 20.7 | 2.5 | 50.5 | 24.5 | **26.8** | +13.6 | 19.5k |
| B2 &mdash; segmentation-only | 34.0 | 10.3 | 0.2 | 18.3 | 7.5 | 35.8 | 12.5 | 0.2 | 21.6 | 7.8 | 34.2 | 10.8 | 0.2 | 20.6 | 7.5 | **15.1** | +1.9 | 25.0k |
| S1 &mdash; protected gradient proj. | 38.5 | 12.2 | 0.2 | 28.8 | 11.8 | 44.5 | 18.7 | 3.2 | 44.3 | 21.0 | 42.5 | 20.2 | 6.2 | 42.5 | 21.5 | **25.2** | +12.0 | 25.0k |
| S2 &mdash; CAGrad consensus | 39.3 | 11.9 | 0.3 | 32.1 | 11.4 | 46.2 | 19.8 | 2.5 | 45.3 | 20.7 | 43.0 | 21.5 | 5.7 | 43.4 | 22.4 | **25.7** | +12.5 | 25.0k |
| S3 &mdash; conflict hard-decouple | 38.9 | 12.5 | 0.2 | 30.1 | 10.7 | 45.4 | 18.6 | 2.1 | 44.3 | 21.0 | 41.5 | 20.1 | 5.9 | 44.0 | 21.7 | **25.0** | +11.8 | 25.0k |
| S4 &mdash; frozen shared trunk | 34.0 | 9.8 | 0.2 | 17.9 | 6.9 | 40.5 | 10.7 | 0.2 | 22.2 | 7.3 | 41.7 | 10.8 | 0.2 | 23.3 | 7.5 | **15.9** | +2.8 | 25.0k |
| S5 &mdash; dynamic aux weight | 37.3 | 11.1 | 0.2 | 26.5 | 10.6 | 45.8 | 17.5 | 2.3 | 45.0 | 18.6 | 41.2 | 18.1 | 4.9 | 42.3 | 18.4 | **23.8** | +10.6 | 25.0k |
| S6 &mdash; seg-head-only routing | 36.7 | 11.4 | 0.2 | 24.9 | 11.0 | 50.4 | 19.3 | 1.2 | 48.4 | 23.6 | 49.0 | 21.0 | 2.2 | 49.4 | 26.3 | **26.7** | +13.6 | 25.0k |

## Table 2
**Cityscapes to Cityscapes-C, long-term &mdash; mIoU.** Five corruptions repeated ten times, no reset (50 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | R1 Fog | R1 Motion | R1 Snow | R1 Bright | R1 Defocus | R5 Fog | R5 Motion | R5 Snow | R5 Bright | R5 Defocus | R10 Fog | R10 Motion | R10 Snow | R10 Bright | R10 Defocus | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Source (no adaptation) | 37.6 | 32.4 | 7.5 | 32.1 | 26.4 | 37.6 | 32.4 | 7.5 | 32.1 | 26.4 | 37.6 | 32.4 | 7.5 | 32.1 | 26.4 | **27.2** | +0.0 | / |
| Ours, full MTL (E11, thr 0.9) | 40.6 | 34.8 | 7.9 | 38.1 | 32.9 | 37.4 | 36.8 | 9.6 | 41.5 | 34.3 | 30.3 | 32.5 | 11.6 | 38.0 | 30.0 | **31.1** | +3.9 | 25.0k |
| Ours, full MTL (E13a, thr 0.8) | 40.4 | 34.7 | 7.9 | 37.7 | 32.2 | 36.8 | 36.6 | 10.9 | 41.6 | 33.3 | 29.6 | 31.8 | 12.8 | 37.3 | 29.0 | **30.8** | +3.6 | 25.0k |
| C1 &mdash; no cross-task losses | 39.4 | 33.8 | 7.6 | 35.2 | 30.9 | 40.2 | 38.2 | 8.2 | 42.4 | 35.4 | 36.3 | 36.6 | 8.4 | 41.5 | 34.3 | **32.0** | +4.7 | 25.0k |
| B1 &mdash; detection-only | 39.7 | 33.9 | 7.6 | 37.6 | 32.2 | 48.1 | 37.6 | 9.5 | 50.8 | 37.2 | 46.8 | 36.5 | 11.1 | 51.9 | 35.4 | **35.5** | +8.3 | 19.5k |
| B2 &mdash; segmentation-only | 38.1 | 33.0 | 7.8 | 33.4 | 29.4 | 35.2 | 35.5 | 8.5 | 34.3 | 30.8 | 30.2 | 34.2 | 8.4 | 32.1 | 28.7 | **28.2** | +1.0 | 25.0k |
| S1 &mdash; protected gradient proj. | 40.3 | 34.8 | 7.9 | 37.8 | 32.5 | 36.1 | 36.1 | 11.5 | 41.7 | 33.2 | 29.9 | 31.9 | 12.7 | 37.9 | 28.7 | **30.9** | +3.7 | 25.0k |
| S2 &mdash; CAGrad consensus | 41.2 | 34.3 | 7.7 | 40.1 | 33.0 | 37.6 | 36.2 | 10.7 | 42.3 | 33.5 | 30.5 | 32.2 | 12.9 | 37.9 | 30.1 | **31.3** | +4.1 | 25.0k |
| S3 &mdash; conflict hard-decouple | 40.5 | 35.1 | 7.8 | 38.4 | 33.2 | 36.7 | 36.7 | 10.0 | 42.1 | 34.3 | 29.7 | 32.3 | 12.5 | 37.7 | 29.5 | **31.0** | +3.8 | 25.0k |
| S4 &mdash; frozen shared trunk | 37.7 | 32.4 | 7.6 | 32.4 | 26.4 | 37.7 | 31.6 | 8.2 | 33.3 | 25.1 | 36.5 | 30.4 | 8.6 | 33.4 | 23.4 | **27.0** | -0.2 | 25.0k |
| S5 &mdash; dynamic aux weight | 39.7 | 33.9 | 7.6 | 37.5 | 31.7 | 37.8 | 35.4 | 10.1 | 42.7 | 32.1 | 30.4 | 30.9 | 12.2 | 38.4 | 27.9 | **30.6** | +3.4 | 25.0k |
| S6 &mdash; seg-head-only routing | 39.6 | 33.9 | 7.6 | 36.7 | 31.8 | 46.8 | 37.2 | 8.9 | 50.1 | 35.8 | 45.6 | 36.2 | 10.3 | 50.8 | 34.3 | **34.7** | +7.5 | 25.0k |

## Table 3
**Cityscapes to ACDC, long-term &mdash; mAP0.5.** Four conditions repeated ten times, no reset (40 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | R1 Fog | R1 Night | R1 Rain | R1 Snow | R4 Fog | R4 Night | R4 Rain | R4 Snow | R7 Fog | R7 Night | R7 Rain | R7 Snow | R10 Fog | R10 Night | R10 Rain | R10 Snow | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AMROD | 52.2 | 16.8 | 37.0 | 37.8 | 55.3 | 21.2 | 37.9 | 40.9 | 56.6 | 22.6 | 39.3 | 41.5 | 56.3 | 23.5 | 38.9 | 42.1 | **39.0** | / | / |
| Ours, full MTL (E11) | 53.9 | 18.2 | 39.4 | 42.0 | 60.0 | 25.4 | 43.3 | 50.0 | 59.9 | 27.0 | 44.0 | 51.1 | 58.3 | 26.7 | 43.6 | 50.8 | **44.0** | / | 16.0k |
| B1 &mdash; detection-only | 52.2 | 17.4 | 38.8 | 39.9 | 57.9 | 24.9 | 44.1 | 46.6 | 59.9 | 27.6 | 45.1 | 48.7 | 61.1 | 27.9 | 44.7 | 48.9 | **43.4** | / | 12.2k |
| S6 &mdash; seg-head-only routing | 52.3 | 17.5 | 38.1 | 40.3 | 58.6 | 25.5 | 43.6 | 46.9 | 60.1 | 27.3 | 44.6 | 47.9 | 60.8 | 28.6 | 43.8 | 48.5 | **43.5** | / | 16.0k |

> Source (no adaptation) has not been measured on ACDC; `scripts/run_source_only_acdc.sh` fills this in (~25 min) and the Gain column needs it.

## Table 4
**Cityscapes to ACDC, long-term &mdash; mIoU.** Four conditions repeated ten times, no reset (40 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | R1 Fog | R1 Night | R1 Rain | R1 Snow | R4 Fog | R4 Night | R4 Rain | R4 Snow | R7 Fog | R7 Night | R7 Rain | R7 Snow | R10 Fog | R10 Night | R10 Rain | R10 Snow | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TENT | 38.5 | 18.7 | 39.4 | 32.3 | 38.8 | 18.3 | 39.4 | 32.4 | 38.7 | 17.7 | 39.1 | 32.2 | 38.4 | 17.2 | 38.7 | 32.0 | **32.0** | / | / |
| CoTTA | 38.3 | 17.9 | 37.8 | 29.6 | 26.2 | 12.9 | 27.0 | 20.3 | 18.9 | 10.1 | 20.9 | 15.8 | 15.7 | 8.6 | 17.9 | 13.6 | **20.2** | / | / |
| Ours, full MTL (E11) | 40.5 | 21.0 | 43.5 | 37.8 | 47.6 | 25.9 | 47.8 | 42.6 | 46.6 | 25.9 | 46.8 | 42.4 | 45.2 | 25.5 | 45.5 | 41.7 | **39.8** | / | 16.0k |
| B1 &mdash; detection-only | 39.3 | 19.8 | 41.3 | 35.4 | 47.5 | 24.1 | 44.9 | 40.3 | 48.3 | 25.9 | 45.5 | 41.2 | 48.2 | 26.3 | 44.6 | 41.0 | **38.9** | / | 12.2k |
| S6 &mdash; seg-head-only routing | 39.2 | 19.8 | 41.3 | 35.3 | 46.1 | 23.7 | 44.2 | 38.9 | 46.7 | 24.7 | 44.6 | 39.8 | 44.6 | 25.2 | 41.9 | 37.5 | **37.6** | / | 16.0k |

> Source (no adaptation) has not been measured on ACDC; `scripts/run_source_only_acdc.sh` fills this in (~25 min) and the Gain column needs it.

## Table 5
**Cityscapes to Cityscapes-C, short-term &mdash; mAP0.5.** Twelve corruptions, single pass (12 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | Defocus | Glass | Motion | Zoom | Snow | Frost | Fog | Bright | Contrast | Elastic | Pixelate | Jpeg | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Source (no adaptation) | 6.6 | 4.0 | 9.5 | 1.1 | 0.2 | 2.1 | 33.0 | 16.4 | 3.7 | 61.5 | 8.2 | 6.0 | **12.7** | +0.0 | / |
| Ours, full MTL (E11) | 8.1 | 6.4 | 12.1 | 2.1 | 0.2 | 5.9 | 42.3 | 31.4 | 9.8 | 57.6 | 13.9 | 14.2 | **17.0** | +4.3 | 6.0k |

## Table 6
**Cityscapes to Cityscapes-C, short-term &mdash; mIoU.** Twelve corruptions, single pass (12 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | Defocus | Glass | Motion | Zoom | Snow | Frost | Fog | Bright | Contrast | Elastic | Pixelate | Jpeg | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Source (no adaptation) | 26.4 | 27.1 | 32.4 | 18.4 | 7.5 | 12.4 | 37.6 | 32.1 | 14.1 | 64.3 | 27.0 | 15.8 | **26.3** | +0.0 | / |
| Ours, full MTL (E11) | 29.1 | 29.3 | 36.4 | 20.3 | 8.1 | 16.2 | 38.6 | 39.0 | 18.8 | 59.8 | 30.2 | 18.7 | **28.7** | +2.5 | 6.0k |

## Table 7
**Cityscapes to ACDC, short-term &mdash; mAP0.5.** Four conditions, single pass (4 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | Fog | Night | Rain | Snow | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|
| Ours, full MTL (E11) | 53.8 | 18.2 | 39.5 | 42.0 | **38.4** | / | 1.6k |
| Ours, CT-CR mode D | 52.5 | 16.6 | 38.0 | 40.7 | **36.9** | / | 1.6k |
| AMROD | 52.1 | 16.9 | 36.8 | 37.8 | **35.9** | / | / |

> Source (no adaptation) has not been measured on ACDC; `scripts/run_source_only_acdc.sh` fills this in (~25 min) and the Gain column needs it.

## Table 8
**Cityscapes to ACDC, short-term &mdash; mIoU.** Four conditions, single pass (4 evaluations). `B`/`C`/`S` rows are single-factor ablations of the reference row above them.

| Condition | Fog | Night | Rain | Snow | Mean | Gain | Iter. |
|---|---|---|---|---|---|---|---|
| Ours, full MTL (E11) | 40.5 | 21.0 | 43.5 | 38.0 | **35.8** | / | 1.6k |
| Ours, CT-CR mode D | 39.8 | 20.0 | 42.9 | 37.4 | **35.1** | / | 1.6k |
| TENT | 38.5 | 18.7 | 39.4 | 32.3 | **32.2** | / | / |
| CoTTA | 38.3 | 17.9 | 37.8 | 29.6 | **30.9** | / | / |

> Source (no adaptation) has not been measured on ACDC; `scripts/run_source_only_acdc.sh` fills this in (~25 min) and the Gain column needs it.

## Factorial ablation (Cityscapes-C long-term, seed 0)

Effect of each task's adaptation loss, all other settings fixed:

| | detection loss ON | detection loss OFF |
|---|---|---|
| **segmentation loss ON** | E13a **24.80** / 30.83 | E21 **15.10** / 28.21 |
| **segmentation loss OFF** | E15 **26.80** / 35.51 | Source **13.17** / 27.21 |

Cells are **mAP0.5** / mIoU. The segmentation loss has almost no main effect but a large negative interaction: it helps slightly on its own and hurts substantially once detection is also adapting.

## Reproducibility and caveats

- **Run-to-run noise.** Adaptation is not deterministic: cuDNN uses non-deterministic convolution backward kernels, and 25k sequential self-training steps with hard pseudo-label thresholds amplify that. Two runs of the identical config and seed differ by up to **1.8 mAP0.5 on a single evaluation**, but only **~0.5 on the 50-evaluation mean** and **~0.02 on mean mIoU**. Treat AP50 differences below 0.5 as ties.

- **Seeds.** Most arms are seed 0 only. Seeds 42/123 exist for E11 (both protocols) and are in progress for E22/E15 on Cityscapes-C.

- **Specialist study.** ST-D (Mask R-CNN) and ST-S (Semantic FPN) change the source checkpoint and are therefore reported separately, never in the same-source tables above.

- **ACDC source row** is missing (see note under the ACDC tables).

## Gaps: runs referenced above that are absent from this machine

- `tent_pfnsrc_cscLT_s0` &mdash; TENT (cscLT)
- `cotta_pfnsrc_cscLT_s0` &mdash; CoTTA (cscLT)
- TENT and CoTTA on Cityscapes-C (both protocols) &mdash; the `csc12` CoTTA log has 0 evaluations; no long-term run exists here
- Source (no adaptation) on ACDC &mdash; `scripts/run_source_only_acdc.sh`, ~25 min
- Seeds 42/123 for E13a on Cityscapes-C long-term &mdash; every `vs full MTL` margin is currently n=1 on the reference
- ST-D / ST-S specialist study &mdash; prepared, blocked on locating the specialist checkpoints on the remote machine

