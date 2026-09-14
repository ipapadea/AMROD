CT-CR A/B/C — Cityscapes-C mixed long-term bundle
=====================================================

Protocol
--------
Cycle:
  fog -> motion_blur -> snow -> brightness -> defocus_blur

Repeat:
  10 rounds, no reset, same continual model instance

Total:
  50 evaluations / variant

Variants
--------
A  V2 + legacy full-box CT-CR + CTPV OFF
B  V2 + hard semantic-support CT-CR, tau=0.3 + CTPV OFF
C  V2 + soft semantic-weighted CT-CR, alpha/floor=0.2 + CTPV OFF

The *_mtl Cityscapes-C registrations are used so the run can emit both
detection and semantic-segmentation metrics.

Cronus path configuration
-------------------------
The scripts default to the gpu1 paths used previously, but EVERY important
path can be overridden without editing the scripts:

  HOST_REPO
  HOST_OUT
  CSC_ROOT
  CITYSCAPES_ROOT
  CITYSCAPES_ANN_ROOT
  DOCKER_IMAGE
  LOG_ROOT

Example:
  export HOST_REPO=/home/ilias/AMROD
  export HOST_OUT=/data/ilias/panoptic_fpn/output
  export CSC_ROOT=/data/vgcmt/datasets/cityscapes_c_amrod
  export CITYSCAPES_ROOT=/data/ilias/cityscapes_pfn
  export CITYSCAPES_ANN_ROOT=/data/vgcmt/datasets/cityscapes/annotations
  export LOG_ROOT=/home/ilias

Preflight:
  bash scripts/preflight_ctcr_mixed_lt.sh

Two-GPU run:
  GPU_AB=2 GPU_C=3 bash scripts/run_ctcr_abc_mixed_lt_parallel.sh

One-GPU batch:
  bash scripts/run_ctcr_abc_mixed_lt_batch.sh 3

Quick summaries:
  python3 scripts/analyze_ctcr_mixed_lt.py /home/ilias/ctcr_A_v2_full_no_ctpv_csc_mixed_lt_x10.log
  python3 scripts/analyze_ctcr_mixed_lt.py /home/ilias/ctcr_B_v2_hard_t03_no_ctpv_csc_mixed_lt_x10.log
  python3 scripts/analyze_ctcr_mixed_lt.py /home/ilias/ctcr_C_v2_soft_a02_no_ctpv_csc_mixed_lt_x10.log

Important
---------
Run the preflight on Cronus before launching a long experiment. It catches the
exact missing Cityscapes annotations mount that caused the previous failure.
