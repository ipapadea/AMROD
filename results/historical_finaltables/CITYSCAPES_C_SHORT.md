# Cityscapes → Cityscapes-C Short

**Protocol:** `12 official corruptions once`  
**Canonical seeds:** `0, 42, 123`  
**Uncertainty:** sample standard deviation (`n-1`, `n=3`).

> Fair-comparison rule: methods should be compared as method ablations only inside the same source-checkpoint section. Values across different source sections are architectural/reference comparisons, not controlled same-source comparisons.

## Source: Mask R-CNN R50-FPN

Checkpoint: `mask_rcnn_R50_cityscapes/model_final.pth`

| Method | n | AP50 | mIoU | R1 AP50 | R10 AP50 | R1 mIoU | R10 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTCMT_Det [historical FT] | 3 | 17.084 ± 0.481 | — | — | — | — | — |

Config:
`detectron2/configs/Cityscapes/ctcmt_det_mr_R_50_ACDC.yaml`

## Source: Semantic-FPN R50

Checkpoint: `semantic_R50_cityscapes/model_final.pth`

| Method | n | AP50 | mIoU | R1 AP50 | R10 AP50 | R1 mIoU | R10 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTCMT_Seg [historical FT] | 3 | — | 28.735 ± 0.051 | — | — | — | — |

Config:
`detectron2/configs/Cityscapes/ctcmt_seg_sf_small_aug_R_50_ACDC.yaml`

## Source: Panoptic-FPN R50

Checkpoint: `panoptic_fpn_R50_cityscapes/model_final.pth`

| Method | n | AP50 | mIoU | R1 AP50 | R10 AP50 | R1 mIoU | R10 mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTCMT-MTL+V2 [historical FT] | 3 | 14.666 ± 0.628 | 27.573 ± 0.147 | — | — | — | — |

Config:
`detectron2/configs/Cityscapes/ctcmt_v2_cross_fisher_pfn_R_50_ACDC.yaml`

## Raw per-seed values

The exact seed-level values used to compute these tables are stored in [`per_seed.csv`](per_seed.csv).
