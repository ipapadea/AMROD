# Cityscapes → Cityscapes-C Long-Term

**Protocol:** `Fog->MotionBlur->Snow->Brightness->DefocusBlur x10`  
**Canonical seeds:** `0, 42, 123`  
**Uncertainty:** sample standard deviation (`n-1`, `n=3`).

> Fair-comparison rule: methods should be compared as method ablations only inside the same source-checkpoint section. Values across different source sections are architectural/reference comparisons, not controlled same-source comparisons.

## Source: Mask R-CNN R50-FPN

Checkpoint: `mask_rcnn_R50_cityscapes/model_final.pth`

| Method | n | AP50 | mIoU | R1 AP50 | R10 AP50 | R1 mIoU | R10 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTCMT_Det [historical FT] | 3 | 21.725 ± 0.045 | — | 18.210 ± 0.391 | 22.623 ± 0.210 | — | — |

Config:
`detectron2/configs/Cityscapes/ctcmt_det_mr_R_50_ACDC.yaml`

## Source: Semantic-FPN R50

Checkpoint: `semantic_R50_cityscapes/model_final.pth`

| Method | n | AP50 | mIoU | R1 AP50 | R10 AP50 | R1 mIoU | R10 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTCMT_Seg [historical FT] | 3 | — | 29.406 ± 0.191 | — | — | 32.177 ± 0.222 | 27.347 ± 0.110 |

Config:
`detectron2/configs/Cityscapes/ctcmt_seg_sf_small_aug_R_50_ACDC.yaml`

## Source: Panoptic-FPN R50

Checkpoint: `panoptic_fpn_R50_cityscapes/model_final.pth`

| Method | n | AP50 | mIoU | R1 AP50 | R10 AP50 | R1 mIoU | R10 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTCMT-MTL+V2 [historical FT] | 3 | 17.792 ± 0.583 | 28.916 ± 0.167 | 16.367 ± 0.395 | 16.646 ± 0.540 | 30.546 ± 0.131 | 25.704 ± 0.379 |

Config:
`detectron2/configs/Cityscapes/ctcmt_v2_cross_fisher_pfn_R_50_ACDC.yaml`

## Raw per-seed values

The exact seed-level values used to compute these tables are stored in [`per_seed.csv`](per_seed.csv).
