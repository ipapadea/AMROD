# Historical FINAL_TABLES benchmark archive

This directory freezes the canonical historical benchmark results collected
before implementing the EXTENSION changes.

## Code provenance

- Branch: `ft-historical-benchmarks`
- Commit: `4dc18a84e75b1b2802f28a9b1c057cc056060ac2`
- Canonical seeds: `0`, `42`, `123`
- Canonical experiments: `27 / 27`
- Detection metric: bbox `AP50`
- Segmentation metric: `mIoU`
- Standard deviation: sample standard deviation (`n-1`)

## Benchmark tables

- [`ACDC_LT.md`](ACDC_LT.md)
- [`CITYSCAPES_C_SHORT.md`](CITYSCAPES_C_SHORT.md)
- [`CITYSCAPES_C_LT.md`](CITYSCAPES_C_LT.md)

Machine-readable results:

- [`per_seed.csv`](per_seed.csv): one row per canonical experiment.
- [`summary.csv`](summary.csv): mean ± sample-std components.

## Fair comparison rule

The exact source checkpoint is part of the experimental condition.

The canonical historical methods use three different source checkpoints:

- `CTCMT_Det [historical FT]`:
  `mask_rcnn_R50_cityscapes/model_final.pth`
- `CTCMT_Seg [historical FT]`:
  `semantic_R50_cityscapes/model_final.pth`
- `CTCMT-MTL+V2 [historical FT]`:
  `panoptic_fpn_R50_cityscapes/model_final.pth`

Therefore, numbers from different source groups may be shown together as
cross-architecture references, but they must not be interpreted as controlled
method-only comparisons.

Future Panoptic-FPN ablations and EXTENSION variants should be compared inside
the Panoptic-FPN source group using the same source checkpoint.

## Validation

All 27 canonical runs have the expected evaluation counts:

- ACDC-LT: 40 evaluations/run.
- Cityscapes-C short: 12 evaluations/run.
- Cityscapes-C LT: 50 evaluations/run.

Seed 0 was additionally checked across all 9 logs with zero tracebacks.

On Cronus, the initial attempts for the following Cityscapes-C runs failed
before evaluation because the Cityscapes COCO annotation JSON was not reachable
at the expected filesystem path:

- seed 123: Det CS-C, Det CS-C-LT, MTL CS-C, MTL CS-C-LT.
- seed 42: Det CS-C, Det CS-C-LT.

The dataset filesystem path was repaired and only those affected experiments
were rerun with the same method configuration, source weights, benchmark,
and seed. The successful reruns are the canonical values recorded here.

## Raw-output locations

Historical raw outputs were produced under:

`/data/ilias/panoptic_fpn/output/ft_historical/`

Seed 0 was run on `gpu1`; seeds 42 and 123 were run on `Cronus`.
