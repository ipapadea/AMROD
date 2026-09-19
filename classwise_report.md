# Class-wise diagnostic study of task interaction under CTTA

Generated 2026-09-19 21:11 by `scripts/classwise_analysis.py` from the run logs. Do not edit by hand - regenerate.

Detection classes are reported as **AP@[.5:.95]**, not AP50: detectron2 prints only the averaged per-category AP, and the per-evaluation prediction files are overwritten by each evaluation so AP50 per class cannot be recomputed offline. Segmentation classes are IoU. Within each benchmark all four conditions share one source checkpoint and one threshold lineage.

## Data coverage

| Benchmark | Condition | log | det evals | seg evals | usable |
|---|---|---|---|---|---|
| Cityscapes-C long-term | Source | `source_only_pfn_cs_c` | 50 | 50 | yes |
| Cityscapes-C long-term | DET | `e15_detonly_thr080_cscLT_s0` | 50 | 50 | yes |
| Cityscapes-C long-term | SEG | `e21_segonly_thr080_cscLT_s0` | 50 | 50 | yes |
| Cityscapes-C long-term | Full | `e13a_thrmax080_cscLT_s0` | 50 | 50 | yes |
| ACDC long-term | Source | `source_only_pfn_acdc_full` | 40 | 40 | yes |
| ACDC long-term | DET | `e25_detonly_acdc_acdcLT_s0` | 40 | 40 | yes |
| ACDC long-term | SEG | `e29_segonly_acdc_acdcLT_s0` | 40 | 40 | yes |
| ACDC long-term | Full | `e11_bothsc_ctcrD_acdcLT_s0` | 40 | 40 | yes |

## Cityscapes-C long-term

### Per-class trajectories, AP

| Class | Cond | Mean | R1 | Peak | R10 | Drift |
|---|---|---|---|---|---|---|
| person | Source | 8.65 | 8.65 | 8.65 (R1) | 8.65 | +0.00 |
| person | DET | 17.32 | 11.33 | 19.00 (R10) | 19.00 | +0.00 |
| person | SEG | 10.25 | 9.29 | 10.49 (R5) | 10.43 | -0.06 |
| person | Full | 17.72 | 12.23 | 19.07 (R9) | 18.96 | -0.11 |
| rider | Source | 8.48 | 8.48 | 8.48 (R1) | 8.48 | +0.00 |
| rider | DET | 17.78 | 11.67 | 19.44 (R10) | 19.44 | +0.00 |
| rider | SEG | 9.90 | 9.21 | 10.19 (R7) | 9.71 | -0.48 |
| rider | Full | 16.42 | 12.55 | 17.71 (R4) | 16.23 | -1.48 |
| car | Source | 18.84 | 18.84 | 18.84 (R1) | 18.84 | +0.00 |
| car | DET | 29.97 | 22.55 | 32.22 (R10) | 32.22 | +0.00 |
| car | SEG | 20.81 | 19.85 | 21.17 (R5) | 20.66 | -0.51 |
| car | Full | 30.59 | 23.30 | 33.19 (R10) | 33.19 | +0.00 |
| truck | Source | 6.18 | 6.18 | 6.18 (R1) | 6.18 | +0.00 |
| truck | DET | 13.43 | 7.35 | 15.89 (R9) | 15.56 | -0.33 |
| truck | SEG | 6.61 | 6.55 | 7.33 (R3) | 5.88 | -1.44 |
| truck | Full | 12.09 | 7.97 | 13.47 (R6) | 12.75 | -0.72 |
| bus | Source | 13.15 | 13.15 | 13.15 (R1) | 13.15 | +0.00 |
| bus | DET | 23.19 | 16.02 | 25.23 (R9) | 24.92 | -0.30 |
| bus | SEG | 16.97 | 14.24 | 17.79 (R5) | 17.55 | -0.24 |
| bus | Full | 20.04 | 17.70 | 23.08 (R3) | 19.05 | -4.03 |
| train | Source | 2.83 | 2.83 | 2.83 (R1) | 2.83 | +0.00 |
| train | DET | 7.24 | 4.55 | 8.94 (R5) | 7.12 | -1.82 |
| train | SEG | 2.90 | 3.26 | 4.11 (R2) | 1.93 | -2.18 |
| train | Full | 3.58 | 4.62 | 4.65 (R2) | 2.95 | -1.70 |
| motorcycle | Source | 2.12 | 2.12 | 2.12 (R1) | 2.12 | +0.00 |
| motorcycle | DET | 6.11 | 2.40 | 7.43 (R10) | 7.43 | +0.00 |
| motorcycle | SEG | 2.40 | 2.12 | 2.69 (R5) | 2.29 | -0.40 |
| motorcycle | Full | 5.08 | 2.53 | 5.93 (R5) | 4.80 | -1.12 |
| bicycle | Source | 6.54 | 6.54 | 6.54 (R1) | 6.54 | +0.00 |
| bicycle | DET | 13.90 | 8.90 | 15.14 (R10) | 15.14 | +0.00 |
| bicycle | SEG | 7.42 | 6.84 | 7.61 (R4) | 7.33 | -0.28 |
| bicycle | Full | 13.76 | 9.38 | 14.84 (R10) | 14.84 | +0.00 |

### Per-class trajectories, IoU

| Class | Cond | Mean | R1 | Peak | R10 | Drift |
|---|---|---|---|---|---|---|
| road | Source | 78.76 | 78.76 | 78.76 (R1) | 78.76 | +0.00 |
| road | DET | 83.46 | 80.47 | 84.58 (R8) | 84.48 | -0.09 |
| road | SEG | 80.23 | 79.55 | 80.48 (R5) | 80.02 | -0.46 |
| road | Full | 81.49 | 80.46 | 82.07 (R3) | 81.50 | -0.57 |
| sidewalk | Source | 32.83 | 32.83 | 32.83 (R1) | 32.83 | +0.00 |
| sidewalk | DET | 40.61 | 36.63 | 41.82 (R5) | 40.02 | -1.81 |
| sidewalk | SEG | 33.63 | 34.11 | 34.75 (R3) | 31.76 | -2.99 |
| sidewalk | Full | 38.46 | 37.09 | 40.50 (R3) | 36.61 | -3.89 |
| building | Source | 56.51 | 56.51 | 56.51 (R1) | 56.51 | +0.00 |
| building | DET | 66.89 | 60.50 | 68.76 (R6) | 67.78 | -0.98 |
| building | SEG | 58.77 | 58.24 | 59.84 (R4) | 57.70 | -2.14 |
| building | Full | 62.69 | 60.67 | 63.70 (R3) | 61.41 | -2.28 |
| wall | Source | 4.94 | 4.94 | 4.94 (R1) | 4.94 | +0.00 |
| wall | DET | 8.61 | 6.33 | 9.78 (R3) | 7.92 | -1.86 |
| wall | SEG | 4.15 | 5.27 | 5.38 (R2) | 2.50 | -2.88 |
| wall | Full | 4.64 | 6.06 | 7.16 (R3) | 1.51 | -5.64 |
| fence | Source | 11.43 | 11.43 | 11.43 (R1) | 11.43 | +0.00 |
| fence | DET | 13.09 | 13.07 | 14.88 (R2) | 12.69 | -2.19 |
| fence | SEG | 10.90 | 11.69 | 11.94 (R3) | 9.44 | -2.50 |
| fence | Full | 13.05 | 13.23 | 15.32 (R3) | 9.57 | -5.75 |
| pole | Source | 18.76 | 18.76 | 18.76 (R1) | 18.76 | +0.00 |
| pole | DET | 27.14 | 21.97 | 28.41 (R5) | 27.80 | -0.61 |
| pole | SEG | 17.05 | 19.52 | 19.52 (R1) | 13.08 | -6.44 |
| pole | Full | 18.12 | 21.02 | 22.71 (R2) | 12.73 | -9.98 |
| traffic light | Source | 8.77 | 8.77 | 8.77 (R1) | 8.77 | +0.00 |
| traffic light | DET | 15.52 | 10.45 | 17.68 (R9) | 17.46 | -0.23 |
| traffic light | SEG | 6.65 | 8.88 | 8.88 (R1) | 3.98 | -4.89 |
| traffic light | Full | 8.18 | 10.15 | 11.83 (R3) | 2.91 | -8.92 |
| traffic sign | Source | 26.75 | 26.75 | 26.75 (R1) | 26.75 | +0.00 |
| traffic sign | DET | 27.60 | 28.17 | 29.32 (R3) | 26.32 | -3.00 |
| traffic sign | SEG | 27.60 | 27.92 | 29.08 (R3) | 25.20 | -3.89 |
| traffic sign | Full | 32.85 | 30.23 | 36.16 (R4) | 28.27 | -7.89 |
| vegetation | Source | 43.01 | 43.01 | 43.01 (R1) | 43.01 | +0.00 |
| vegetation | DET | 60.21 | 48.58 | 64.07 (R6) | 61.55 | -2.52 |
| vegetation | SEG | 44.68 | 46.08 | 48.11 (R2) | 40.82 | -7.29 |
| vegetation | Full | 47.80 | 47.78 | 51.53 (R3) | 41.93 | -9.60 |
| terrain | Source | 13.17 | 13.17 | 13.17 (R1) | 13.17 | +0.00 |
| terrain | DET | 18.80 | 14.96 | 20.02 (R6) | 18.92 | -1.10 |
| terrain | SEG | 15.08 | 14.33 | 15.39 (R4) | 14.93 | -0.46 |
| terrain | Full | 17.02 | 15.56 | 17.87 (R3) | 15.96 | -1.90 |
| sky | Source | 65.87 | 65.87 | 65.87 (R1) | 65.87 | +0.00 |
| sky | DET | 72.40 | 70.07 | 74.63 (R4) | 69.72 | -4.92 |
| sky | SEG | 68.29 | 68.18 | 68.73 (R4) | 67.91 | -0.82 |
| sky | Full | 73.19 | 70.31 | 73.76 (R6) | 73.39 | -0.37 |
| person | Source | 34.02 | 34.02 | 34.02 (R1) | 34.02 | +0.00 |
| person | DET | 48.41 | 39.37 | 50.24 (R8) | 50.19 | -0.05 |
| person | SEG | 37.40 | 35.69 | 38.20 (R5) | 36.88 | -1.32 |
| person | Full | 41.68 | 40.64 | 45.73 (R3) | 37.01 | -8.72 |
| rider | Source | 7.92 | 7.92 | 7.92 (R1) | 7.92 | +0.00 |
| rider | DET | 23.32 | 11.22 | 28.72 (R10) | 28.72 | +0.00 |
| rider | SEG | 7.89 | 8.79 | 9.54 (R2) | 5.79 | -3.75 |
| rider | Full | 14.30 | 13.61 | 18.12 (R3) | 9.70 | -8.41 |
| car | Source | 57.28 | 57.28 | 57.28 (R1) | 57.28 | +0.00 |
| car | DET | 63.01 | 59.51 | 63.93 (R6) | 62.97 | -0.96 |
| car | SEG | 59.33 | 58.23 | 59.79 (R4) | 59.00 | -0.78 |
| car | Full | 60.32 | 60.09 | 62.99 (R2) | 57.51 | -5.48 |
| truck | Source | 8.28 | 8.28 | 8.28 (R1) | 8.28 | +0.00 |
| truck | DET | 16.28 | 10.22 | 17.58 (R5) | 16.93 | -0.65 |
| truck | SEG | 10.25 | 9.16 | 11.12 (R4) | 9.45 | -1.67 |
| truck | Full | 10.77 | 11.08 | 13.21 (R2) | 9.29 | -3.92 |
| bus | Source | 16.38 | 16.38 | 16.38 (R1) | 16.38 | +0.00 |
| bus | DET | 27.93 | 20.11 | 32.43 (R10) | 32.43 | +0.00 |
| bus | SEG | 22.69 | 18.56 | 23.96 (R5) | 22.93 | -1.02 |
| bus | Full | 17.43 | 20.93 | 23.35 (R3) | 13.91 | -9.44 |
| train | Source | 5.65 | 5.65 | 5.65 (R1) | 5.65 | +0.00 |
| train | DET | 12.08 | 7.23 | 13.42 (R6) | 11.91 | -1.52 |
| train | SEG | 5.66 | 6.14 | 7.10 (R2) | 4.02 | -3.08 |
| train | Full | 4.28 | 7.95 | 7.95 (R1) | 2.41 | -5.55 |
| motorcycle | Source | 4.67 | 4.67 | 4.67 (R1) | 4.67 | +0.00 |
| motorcycle | DET | 8.72 | 5.51 | 9.54 (R3) | 9.11 | -0.43 |
| motorcycle | SEG | 4.56 | 5.16 | 5.36 (R2) | 3.59 | -1.77 |
| motorcycle | Full | 2.48 | 5.27 | 6.20 (R2) | 0.04 | -6.16 |
| bicycle | Source | 22.02 | 22.02 | 22.02 (R1) | 22.02 | +0.00 |
| bicycle | DET | 40.70 | 29.61 | 43.57 (R8) | 43.37 | -0.20 |
| bicycle | SEG | 21.11 | 22.71 | 22.72 (R2) | 18.57 | -4.14 |
| bicycle | Full | 37.04 | 28.96 | 38.97 (R6) | 37.94 | -1.03 |

### Class-wise adaptation effects, AP

| Class | Source | d_SEG | d_DET | d_SEG&#124;DET | d_DET&#124;SEG | I_c |
|---|---|---|---|---|---|---|
| person | 8.65 | +1.60 | +8.67 | +0.40 | +7.47 | -1.20 **(neg)** |
| rider | 8.48 | +1.42 | +9.30 | -1.36 | +6.52 | -2.78 **(neg)** |
| car | 18.84 | +1.98 | +11.13 | +0.62 | +9.77 | -1.36 **(neg)** |
| truck | 6.18 | +0.42 | +7.24 | -1.34 | +5.48 | -1.76 **(neg)** |
| bus | 13.15 | +3.83 | +10.04 | -3.15 | +3.06 | -6.98 **(neg)** |
| train | 2.83 | +0.07 | +4.41 | -3.66 | +0.68 | -3.73 **(neg)** |
| motorcycle | 2.12 | +0.27 | +3.98 | -1.02 | +2.69 | -1.29 **(neg)** |
| bicycle | 6.54 | +0.88 | +7.36 | -0.14 | +6.34 | -1.02 **(neg)** |

#### Difficulty vs harmfulness, AP

| Relationship | Spearman rho |
|---|---|
| source score vs I_c | -0.10 |
| full-MTL drift vs I_c | 0.85 |
| source score vs full-MTL drift | 0.22 |

A strongly negative `source score vs I_c` would mean the weakest source classes are the most harmful; near zero means harmfulness is not explained by class difficulty alone.

### Class-wise adaptation effects, IoU

| Class | Source | d_SEG | d_DET | d_SEG&#124;DET | d_DET&#124;SEG | I_c |
|---|---|---|---|---|---|---|
| road | 78.76 | +1.47 | +4.70 | -1.98 | +1.26 | -3.45 **(neg)** |
| sidewalk | 32.83 | +0.80 | +7.78 | -2.15 | +4.83 | -2.95 **(neg)** |
| building | 56.51 | +2.26 | +10.38 | -4.20 | +3.92 | -6.46 **(neg)** |
| wall | 4.94 | -0.79 | +3.67 | -3.97 | +0.49 | -3.17 **(neg)** |
| fence | 11.43 | -0.53 | +1.66 | -0.05 | +2.15 | +0.48 |
| pole | 18.76 | -1.70 | +8.38 | -9.02 | +1.06 | -7.32 **(neg)** |
| traffic light | 8.77 | -2.12 | +6.75 | -7.34 | +1.54 | -5.22 **(neg)** |
| traffic sign | 26.75 | +0.86 | +0.86 | +5.24 | +5.24 | +4.39 |
| vegetation | 43.01 | +1.67 | +17.20 | -12.41 | +3.12 | -14.08 **(neg)** |
| terrain | 13.17 | +1.90 | +5.62 | -1.78 | +1.94 | -3.69 **(neg)** |
| sky | 65.87 | +2.42 | +6.52 | +0.79 | +4.90 | -1.62 **(neg)** |
| person | 34.02 | +3.38 | +14.38 | -6.73 | +4.27 | -10.11 **(neg)** |
| rider | 7.92 | -0.03 | +15.40 | -9.02 | +6.41 | -8.99 **(neg)** |
| car | 57.28 | +2.06 | +5.73 | -2.69 | +0.99 | -4.74 **(neg)** |
| truck | 8.28 | +1.97 | +8.00 | -5.51 | +0.52 | -7.48 **(neg)** |
| bus | 16.38 | +6.31 | +11.55 | -10.50 | -5.25 | -16.81 **(neg)** |
| train | 5.65 | +0.00 | +6.43 | -7.80 | -1.38 | -7.80 **(neg)** |
| motorcycle | 4.67 | -0.11 | +4.05 | -6.24 | -2.08 | -6.13 **(neg)** |
| bicycle | 22.02 | -0.90 | +18.68 | -3.66 | +15.93 | -2.75 **(neg)** |

#### Thing classes (mapped to detection) vs stuff classes

| Group | n | mean I_c | mean d_SEG&#124;DET | n with I_c<0 |
|---|---|---|---|---|
| thing (mapped) | 8 | -8.10 | -6.52 | 8/8 |
| stuff (seg-only) | 11 | -3.92 | -3.35 | 9/11 |

#### Difficulty vs harmfulness, IoU

| Relationship | Spearman rho |
|---|---|
| source score vs I_c | 0.17 |
| full-MTL drift vs I_c | 0.55 |
| source score vs full-MTL drift | 0.32 |

A strongly negative `source score vs I_c` would mean the weakest source classes are the most harmful; near zero means harmfulness is not explained by class difficulty alone.

## ACDC long-term

### Per-class trajectories, AP

| Class | Cond | Mean | R1 | Peak | R10 | Drift |
|---|---|---|---|---|---|---|
| person | Source | 24.60 | 24.60 | 24.60 (R1) | 24.60 | +0.00 |
| person | DET | 30.47 | 26.41 | 32.19 (R10) | 32.19 | +0.00 |
| person | SEG | 26.34 | 25.32 | 26.86 (R10) | 26.86 | +0.00 |
| person | Full | 31.56 | 27.25 | 33.01 (R10) | 33.01 | +0.00 |
| rider | Source | 20.08 | 20.08 | 20.08 (R1) | 20.08 | +0.00 |
| rider | DET | 27.04 | 21.37 | 29.06 (R10) | 29.06 | +0.00 |
| rider | SEG | 23.09 | 20.75 | 24.13 (R10) | 24.13 | +0.00 |
| rider | Full | 27.97 | 23.00 | 29.40 (R7) | 29.14 | -0.26 |
| car | Source | 43.49 | 43.49 | 43.49 (R1) | 43.49 | +0.00 |
| car | DET | 51.04 | 45.88 | 52.88 (R10) | 52.88 | +0.00 |
| car | SEG | 45.90 | 44.28 | 46.43 (R10) | 46.43 | +0.00 |
| car | Full | 51.73 | 46.74 | 53.32 (R10) | 53.32 | +0.00 |
| truck | Source | 16.16 | 16.16 | 16.16 (R1) | 16.16 | +0.00 |
| truck | DET | 20.24 | 17.27 | 21.21 (R6) | 20.90 | -0.31 |
| truck | SEG | 17.92 | 16.69 | 18.40 (R8) | 18.33 | -0.07 |
| truck | Full | 18.39 | 17.02 | 18.95 (R6) | 17.97 | -0.98 |
| bus | Source | 17.16 | 17.16 | 17.16 (R1) | 17.16 | +0.00 |
| bus | DET | 21.77 | 18.27 | 23.89 (R10) | 23.89 | +0.00 |
| bus | SEG | 19.94 | 17.34 | 21.04 (R9) | 20.85 | -0.20 |
| bus | Full | 23.47 | 19.24 | 24.36 (R6) | 23.32 | -1.04 |
| train | Source | 13.33 | 13.33 | 13.33 (R1) | 13.33 | +0.00 |
| train | DET | 16.65 | 13.48 | 18.13 (R8) | 17.47 | -0.66 |
| train | SEG | 14.05 | 13.25 | 14.54 (R9) | 14.38 | -0.16 |
| train | Full | 14.37 | 13.53 | 15.14 (R6) | 13.39 | -1.75 |
| motorcycle | Source | 11.81 | 11.81 | 11.81 (R1) | 11.81 | +0.00 |
| motorcycle | DET | 18.47 | 14.88 | 20.69 (R9) | 20.37 | -0.32 |
| motorcycle | SEG | 13.01 | 12.37 | 13.65 (R10) | 13.65 | +0.00 |
| motorcycle | Full | 19.62 | 16.11 | 21.25 (R7) | 20.81 | -0.45 |
| bicycle | Source | 13.59 | 13.59 | 13.59 (R1) | 13.59 | +0.00 |
| bicycle | DET | 21.62 | 16.05 | 23.36 (R7) | 23.26 | -0.09 |
| bicycle | SEG | 16.84 | 14.58 | 17.58 (R5) | 17.46 | -0.12 |
| bicycle | Full | 21.91 | 16.25 | 23.24 (R8) | 22.82 | -0.43 |

### Per-class trajectories, IoU

| Class | Cond | Mean | R1 | Peak | R10 | Drift |
|---|---|---|---|---|---|---|
| road | Source | 58.23 | 58.23 | 58.23 (R1) | 58.23 | +0.00 |
| road | DET | 71.17 | 60.46 | 74.79 (R10) | 74.79 | +0.00 |
| road | SEG | 64.76 | 60.06 | 67.98 (R10) | 67.98 | +0.00 |
| road | Full | 69.53 | 62.93 | 72.78 (R10) | 72.78 | +0.00 |
| sidewalk | Source | 28.14 | 28.14 | 28.14 (R1) | 28.14 | +0.00 |
| sidewalk | DET | 31.97 | 29.64 | 33.17 (R9) | 32.60 | -0.57 |
| sidewalk | SEG | 30.11 | 28.87 | 30.66 (R8) | 30.31 | -0.35 |
| sidewalk | Full | 36.26 | 30.38 | 38.43 (R9) | 38.17 | -0.26 |
| building | Source | 59.93 | 59.93 | 59.93 (R1) | 59.93 | +0.00 |
| building | DET | 67.70 | 62.58 | 68.98 (R7) | 68.82 | -0.16 |
| building | SEG | 63.88 | 62.32 | 64.57 (R5) | 63.36 | -1.21 |
| building | Full | 67.71 | 63.35 | 68.90 (R6) | 68.57 | -0.32 |
| wall | Source | 9.11 | 9.11 | 9.11 (R1) | 9.11 | +0.00 |
| wall | DET | 13.21 | 10.39 | 14.12 (R8) | 13.55 | -0.57 |
| wall | SEG | 10.10 | 9.73 | 10.33 (R6) | 9.75 | -0.58 |
| wall | Full | 15.83 | 11.93 | 17.16 (R10) | 17.16 | +0.00 |
| fence | Source | 11.09 | 11.09 | 11.09 (R1) | 11.09 | +0.00 |
| fence | DET | 15.56 | 11.99 | 17.04 (R5) | 16.41 | -0.63 |
| fence | SEG | 13.78 | 11.94 | 14.47 (R9) | 14.37 | -0.11 |
| fence | Full | 12.90 | 12.10 | 13.75 (R4) | 11.77 | -1.99 |
| pole | Source | 32.44 | 32.44 | 32.44 (R1) | 32.44 | +0.00 |
| pole | DET | 33.88 | 32.95 | 34.34 (R7) | 34.20 | -0.15 |
| pole | SEG | 35.34 | 33.59 | 35.88 (R5) | 35.37 | -0.52 |
| pole | Full | 37.00 | 34.02 | 37.94 (R5) | 37.11 | -0.83 |
| traffic light | Source | 34.38 | 34.38 | 34.38 (R1) | 34.38 | +0.00 |
| traffic light | DET | 36.43 | 35.02 | 37.34 (R7) | 37.19 | -0.15 |
| traffic light | SEG | 37.66 | 35.68 | 38.23 (R5) | 38.02 | -0.21 |
| traffic light | Full | 44.68 | 37.26 | 47.55 (R10) | 47.55 | +0.00 |
| traffic sign | Source | 38.42 | 38.42 | 38.42 (R1) | 38.42 | +0.00 |
| traffic sign | DET | 43.59 | 39.65 | 44.86 (R7) | 44.72 | -0.14 |
| traffic sign | SEG | 41.33 | 39.57 | 41.84 (R5) | 41.45 | -0.39 |
| traffic sign | Full | 45.04 | 40.67 | 46.37 (R10) | 46.37 | +0.00 |
| vegetation | Source | 52.30 | 52.30 | 52.30 (R1) | 52.30 | +0.00 |
| vegetation | DET | 64.15 | 55.71 | 67.28 (R9) | 67.24 | -0.03 |
| vegetation | SEG | 54.03 | 53.66 | 54.23 (R6) | 54.01 | -0.21 |
| vegetation | Full | 60.38 | 56.38 | 61.57 (R10) | 61.57 | +0.00 |
| terrain | Source | 20.12 | 20.12 | 20.12 (R1) | 20.12 | +0.00 |
| terrain | DET | 21.66 | 20.53 | 22.30 (R9) | 22.05 | -0.25 |
| terrain | SEG | 21.84 | 20.29 | 23.03 (R10) | 23.03 | +0.00 |
| terrain | Full | 23.40 | 21.02 | 24.15 (R8) | 24.05 | -0.10 |
| sky | Source | 60.42 | 60.42 | 60.42 (R1) | 60.42 | +0.00 |
| sky | DET | 70.74 | 63.94 | 71.93 (R10) | 71.93 | +0.00 |
| sky | SEG | 67.10 | 63.67 | 69.21 (R10) | 69.21 | +0.00 |
| sky | Full | 68.31 | 64.46 | 70.65 (R10) | 70.65 | +0.00 |
| person | Source | 25.68 | 25.68 | 25.68 (R1) | 25.68 | +0.00 |
| person | DET | 38.65 | 29.65 | 41.94 (R10) | 41.94 | +0.00 |
| person | SEG | 33.24 | 28.15 | 36.63 (R10) | 36.63 | +0.00 |
| person | Full | 34.42 | 36.27 | 41.63 (R2) | 27.05 | -14.58 |
| rider | Source | 34.91 | 34.91 | 34.91 (R1) | 34.91 | +0.00 |
| rider | DET | 42.72 | 35.81 | 45.71 (R10) | 45.71 | +0.00 |
| rider | SEG | 36.31 | 35.65 | 36.73 (R2) | 36.34 | -0.39 |
| rider | Full | 42.81 | 36.01 | 44.47 (R6) | 43.50 | -0.98 |
| car | Source | 62.10 | 62.10 | 62.10 (R1) | 62.10 | +0.00 |
| car | DET | 70.77 | 64.14 | 72.46 (R7) | 72.05 | -0.42 |
| car | SEG | 68.37 | 63.74 | 70.37 (R10) | 70.37 | +0.00 |
| car | Full | 64.33 | 66.70 | 70.04 (R3) | 57.08 | -12.95 |
| truck | Source | 11.85 | 11.85 | 11.85 (R1) | 11.85 | +0.00 |
| truck | DET | 17.12 | 13.40 | 18.17 (R7) | 17.43 | -0.74 |
| truck | SEG | 16.60 | 13.02 | 18.33 (R10) | 18.33 | +0.00 |
| truck | Full | 16.09 | 14.16 | 19.24 (R4) | 12.52 | -6.72 |
| bus | Source | 11.16 | 11.16 | 11.16 (R1) | 11.16 | +0.00 |
| bus | DET | 14.40 | 11.85 | 15.93 (R8) | 14.90 | -1.03 |
| bus | SEG | 13.60 | 11.58 | 15.02 (R8) | 14.62 | -0.41 |
| bus | Full | 14.78 | 13.22 | 17.13 (R3) | 12.82 | -4.32 |
| train | Source | 19.36 | 19.36 | 19.36 (R1) | 19.36 | +0.00 |
| train | DET | 29.57 | 22.02 | 32.72 (R7) | 28.31 | -4.41 |
| train | SEG | 28.89 | 22.09 | 31.48 (R10) | 31.48 | +0.00 |
| train | Full | 35.59 | 25.95 | 38.78 (R6) | 35.33 | -3.45 |
| motorcycle | Source | 15.72 | 15.72 | 15.72 (R1) | 15.72 | +0.00 |
| motorcycle | DET | 23.02 | 17.42 | 24.53 (R10) | 24.53 | +0.00 |
| motorcycle | SEG | 20.19 | 16.82 | 22.14 (R10) | 22.14 | +0.00 |
| motorcycle | Full | 29.10 | 19.42 | 31.65 (R6) | 29.76 | -1.88 |
| bicycle | Source | 26.47 | 26.47 | 26.47 (R1) | 26.47 | +0.00 |
| bicycle | DET | 31.85 | 28.11 | 33.20 (R4) | 31.93 | -1.27 |
| bicycle | SEG | 34.12 | 28.68 | 35.83 (R8) | 35.45 | -0.37 |
| bicycle | Full | 38.04 | 32.20 | 40.37 (R4) | 36.09 | -4.28 |

### Class-wise adaptation effects, AP

| Class | Source | d_SEG | d_DET | d_SEG&#124;DET | d_DET&#124;SEG | I_c |
|---|---|---|---|---|---|---|
| person | 24.60 | +1.74 | +5.87 | +1.09 | +5.22 | -0.65 **(neg)** |
| rider | 20.08 | +3.02 | +6.96 | +0.94 | +4.88 | -2.08 **(neg)** |
| car | 43.49 | +2.41 | +7.55 | +0.69 | +5.83 | -1.72 **(neg)** |
| truck | 16.16 | +1.75 | +4.07 | -1.84 | +0.48 | -3.60 **(neg)** |
| bus | 17.16 | +2.78 | +4.61 | +1.71 | +3.54 | -1.07 **(neg)** |
| train | 13.33 | +0.72 | +3.32 | -2.29 | +0.32 | -3.00 **(neg)** |
| motorcycle | 11.81 | +1.20 | +6.67 | +1.15 | +6.62 | -0.05 **(neg)** |
| bicycle | 13.59 | +3.25 | +8.02 | +0.30 | +5.07 | -2.95 **(neg)** |

#### Difficulty vs harmfulness, AP

| Relationship | Spearman rho |
|---|---|
| source score vs I_c | 0.14 |
| full-MTL drift vs I_c | 0.38 |
| source score vs full-MTL drift | 0.71 |

A strongly negative `source score vs I_c` would mean the weakest source classes are the most harmful; near zero means harmfulness is not explained by class difficulty alone.

### Class-wise adaptation effects, IoU

| Class | Source | d_SEG | d_DET | d_SEG&#124;DET | d_DET&#124;SEG | I_c |
|---|---|---|---|---|---|---|
| road | 58.23 | +6.52 | +12.94 | -1.64 | +4.78 | -8.16 **(neg)** |
| sidewalk | 28.14 | +1.97 | +3.83 | +4.29 | +6.15 | +2.32 |
| building | 59.93 | +3.95 | +7.77 | +0.02 | +3.84 | -3.93 **(neg)** |
| wall | 9.11 | +0.99 | +4.10 | +2.62 | +5.73 | +1.63 |
| fence | 11.09 | +2.69 | +4.48 | -2.66 | -0.88 | -5.36 **(neg)** |
| pole | 32.44 | +2.90 | +1.44 | +3.13 | +1.66 | +0.23 |
| traffic light | 34.38 | +3.29 | +2.05 | +8.26 | +7.02 | +4.97 |
| traffic sign | 38.42 | +2.90 | +5.16 | +1.45 | +3.71 | -1.46 **(neg)** |
| vegetation | 52.30 | +1.73 | +11.84 | -3.77 | +6.34 | -5.50 **(neg)** |
| terrain | 20.12 | +1.72 | +1.54 | +1.74 | +1.56 | +0.02 |
| sky | 60.42 | +6.67 | +10.32 | -2.43 | +1.21 | -9.11 **(neg)** |
| person | 25.68 | +7.55 | +12.96 | -4.22 | +1.19 | -11.78 **(neg)** |
| rider | 34.91 | +1.40 | +7.81 | +0.09 | +6.50 | -1.31 **(neg)** |
| car | 62.10 | +6.27 | +8.67 | -6.44 | -4.04 | -12.71 **(neg)** |
| truck | 11.85 | +4.75 | +5.27 | -1.03 | -0.51 | -5.78 **(neg)** |
| bus | 11.16 | +2.44 | +3.24 | +0.37 | +1.18 | -2.07 **(neg)** |
| train | 19.36 | +9.54 | +10.22 | +6.02 | +6.70 | -3.52 **(neg)** |
| motorcycle | 15.72 | +4.47 | +7.30 | +6.08 | +8.91 | +1.61 |
| bicycle | 26.47 | +7.66 | +5.39 | +6.19 | +3.92 | -1.47 **(neg)** |

#### Thing classes (mapped to detection) vs stuff classes

| Group | n | mean I_c | mean d_SEG&#124;DET | n with I_c<0 |
|---|---|---|---|---|
| thing (mapped) | 8 | -4.63 | +0.88 | 7/8 |
| stuff (seg-only) | 11 | -2.21 | +1.00 | 6/11 |

#### Difficulty vs harmfulness, IoU

| Relationship | Spearman rho |
|---|---|
| source score vs I_c | -0.35 |
| full-MTL drift vs I_c | 0.36 |
| source score vs full-MTL drift | 0.33 |

A strongly negative `source score vs I_c` would mean the weakest source classes are the most harmful; near zero means harmfulness is not explained by class difficulty alone.

## Cross-benchmark sign flip

### AP

| Class | I_c ACDC | I_c CSC | verdict |
|---|---|---|---|
| person | -0.65 | -1.20 | harmful on both |
| rider | -2.08 | -2.78 | harmful on both |
| car | -1.72 | -1.36 | harmful on both |
| truck | -3.60 | -1.76 | harmful on both |
| bus | -1.07 | -6.98 | harmful on both |
| train | -3.00 | -3.73 | harmful on both |
| motorcycle | -0.05 | -1.29 | harmful on both |
| bicycle | -2.95 | -1.02 | harmful on both |

### IoU

| Class | I_c ACDC | I_c CSC | verdict |
|---|---|---|---|
| road | -8.16 | -3.45 | harmful on both |
| sidewalk | +2.32 | -2.95 | **flips: helps on ACDC, harms on CSC** |
| building | -3.93 | -6.46 | harmful on both |
| wall | +1.63 | -3.17 | **flips: helps on ACDC, harms on CSC** |
| fence | -5.36 | +0.48 | flips the other way |
| pole | +0.23 | -7.32 | **flips: helps on ACDC, harms on CSC** |
| traffic light | +4.97 | -5.22 | **flips: helps on ACDC, harms on CSC** |
| traffic sign | -1.46 | +4.39 | flips the other way |
| vegetation | -5.50 | -14.08 | harmful on both |
| terrain | +0.02 | -3.69 | **flips: helps on ACDC, harms on CSC** |
| sky | -9.11 | -1.62 | harmful on both |
| person | -11.78 | -10.11 | harmful on both |
| rider | -1.31 | -8.99 | harmful on both |
| car | -12.71 | -4.74 | harmful on both |
| truck | -5.78 | -7.48 | harmful on both |
| bus | -2.07 | -16.81 | harmful on both |
| train | -3.52 | -7.80 | harmful on both |
| motorcycle | +1.61 | -6.13 | **flips: helps on ACDC, harms on CSC** |
| bicycle | -1.47 | -2.75 | harmful on both |

## Not answerable from the current logs

The following require new instrumentation and a re-run; they are not recoverable from the existing logs.

- **Per-class detection pseudo-label dynamics** (per-class threshold, accepted count, mean teacher confidence, pseudo-label precision/recall). The adapter logs only aggregate `n_pseudo` and `thr=min/mean/max` over all classes.
- **Per-class semantic pseudo-label reliability** (teacher entropy, confidence, predicted vs GT pixel frequency, per-class loss contribution). None of these are emitted.
- **Per-class AP50.** Only per-category AP@[.5:.95] is printed, and the prediction dumps are overwritten each evaluation.

