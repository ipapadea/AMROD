# CTTA Experiment Guide

Quick reference for running all CTTA experiments in this workspace.

## Repository Layout

```
AMROD/                         ← detectron2-based CTTA (Panoptic FPN / Mask R-CNN)
  scripts/
    run_ctta_acdc.sh           ← low-level runner: GPU  TRACK  [REPEATS]
    run_d2_ctta.sh             ← self-service wrapper (see §D2)
    run_cityscapes_c_batch.sh  ← all 5 CS-C methods in one shot
    run_longtask_batch.sh      ← 10-round long-term batch (4 methods)
  detectron2/configs/Cityscapes/
    amrod_mask_rcnn_R_50_ACDC.yaml
    ctcmt_v4b_proto_pfn_R_50_ACDC.yaml   ← best ACDC config
    amrod_mask_rcnn_R_50_CS_C.yaml
    ctcmt_mtl_R_50_CS_C.yaml             ← best CS-C config
    ctcmt_det_shift_R_50_CTTA.yaml
    amrod_shift_R_50_CTTA.yaml
  tools/harvest_results.py    ← parse logs → results/registry.json
  results/registry.json       ← persistent metrics store

TriLiteNet/                    ← lightweight MTL model (ESPNet backbone, 2.3M params)
  run_trilit_ctta.sh           ← self-service wrapper (see §TriLiteNet)
  tools/
    train_cityscapes.py        ← source model training
    adapt_ctta.py              ← CTTA evaluation driver
    eval_cityscapes.py         ← source model evaluation
  lib/
    adapters/
      cotta.py                 ← CoTTA segmentation adapter
      amrod_mtl.py             ← AMROD-style det+seg adapter
      ctcmt_mtl.py             ← CT-CMT (CTCL + proto + Fisher restore)
    dataset/
      cityscapes.py            ← training data loader
      acdc.py                  ← CTTA evaluation stream (seg GT + det GT)
    models/
      TriLiteCityscapes.py     ← det+seg MTL model definition
```

---

## Source Models (Panoptic FPN / Mask R-CNN)

| Model | Path (inside container) | Task |
|:---|:---|:---|
| Panoptic FPN | `/workspace/output/panoptic_fpn_R50_cityscapes/model_final.pth` | det+seg |
| Mask R-CNN | `/workspace/output/mask_rcnn_R50_cityscapes/model_final.pth` | det-only |
| Semantic FPN | `/workspace/output/semantic_R50_cityscapes/model_final.pth` | seg-only |
| AMROD official | `/workspace/weights/cityscapes_train_final.pth` | det-only |

## Source Models (TriLiteNet)

| Model | Checkpoint | Best metric |
|:---|:---|:---|
| MTL (det+seg) | `runs/cityscapes_mtl_base_native/ckpt_best_det.pth` | mIoU=62.2%, mAP50=45.3% |
| Det-only | `runs/cityscapes_det_base_native/ckpt_best_det.pth` | mAP50=44.7% |
| Seg-only | `runs/cityscapes_seg_base_native/ckpt_best_seg.pth` | mIoU=57.97% |

---

## §D2 — Detectron2 Experiments

### Self-service (recommended)

```bash
# Syntax
bash scripts/run_d2_ctta.sh --gpu GPU --method METHOD --bench BENCH [--repeats N]

# Methods:  amrod | amrod-off | ctcmt-det | ctcmt-seg | ctcmt-mtl | cotta
# Benches:  acdc | cs-c | shift | foggy
# Repeats:  1 (short-task, default) | 10 (long-term Table 5)

# Examples
bash scripts/run_d2_ctta.sh --gpu 0 --method ctcmt-mtl --bench acdc
bash scripts/run_d2_ctta.sh --gpu 1 --method amrod     --bench cs-c
bash scripts/run_d2_ctta.sh --gpu 2 --method ctcmt-det --bench acdc --repeats 10
```

### Batch runners

```bash
# Cityscapes-C — all 5 methods sequentially on one GPU
bash scripts/run_cityscapes_c_batch.sh 0

# Long-term CTTA — 3 d2 methods sequentially + TriLiteNet (separate)
bash scripts/run_longtask_batch.sh d2      # GPU 2, sequential
bash scripts/run_longtask_batch.sh trilit  # GPU 3, needs ckpt_best_det.pth
```

### Low-level (direct track name)

```bash
# bash scripts/run_ctta_acdc.sh  GPU  TRACK  [NUM_REPEATS]
bash scripts/run_ctta_acdc.sh 0 ctcmt_v4b        # ACDC 1-round
bash scripts/run_ctta_acdc.sh 0 amrod_shift       # SHIFT
bash scripts/run_ctta_acdc.sh 0 ctcmt_mtl_cs_c   # Cityscapes-C
bash scripts/run_ctta_acdc.sh 0 amrod 10          # ACDC 10-round long-term

# Available tracks (ACDC):     amrod | ctcmt_v4b | ctcmt_det | ctcmt_seg | cotta_semseg
# Available tracks (CS-C):     amrod_cs_c | amrod_official_cs_c | ctcmt_mtl_cs_c
#                               ctcmt_det_cs_c | ctcmt_seg_cs_c | ctcmt_det_mr_cs_c
# Available tracks (SHIFT):    amrod_shift | ctcmt_det_shift
# Available tracks (Foggy CS): amrod_foggy | ctcmt_v2_foggy
```

---

## §TriLiteNet — Lightweight MTL Experiments

### Self-service (recommended)

```bash
# Syntax (run from /home/ilias/TriLiteNet)
bash run_trilit_ctta.sh --gpu GPU --model MODEL --method METHOD [--repeats N] [--limit N]

# Models:   mtl | det | seg
# Methods:  source | cotta | amrod | ctcmt | ctcmt-det | ctcmt-seg
# Repeats:  1 (default) | 10 (long-term)

# Examples
bash run_trilit_ctta.sh --gpu 0 --model mtl --method ctcmt            # CT-CMT-MTL
bash run_trilit_ctta.sh --gpu 1 --model det --method ctcmt-det        # CTCMT_Det
bash run_trilit_ctta.sh --gpu 2 --model seg --method ctcmt-seg        # CTCMT_Seg
bash run_trilit_ctta.sh --gpu 3 --model mtl --method source           # source baseline
bash run_trilit_ctta.sh --gpu 0 --model mtl --method ctcmt --repeats 10  # long-term
bash run_trilit_ctta.sh --gpu 0 --model mtl --method ctcmt --limit 100   # smoke test
```

### Method–model mapping to paper rows

| Paper row | --model | --method |
|:---|:---:|:---:|
| Source-only | mtl | source |
| CoTTA (seg) | seg | cotta |
| AMROD (det) | det | amrod |
| CTCMT_Det | det | ctcmt-det |
| CTCMT_Seg | seg | ctcmt-seg |
| CT-CMT-MTL (ours) | mtl | ctcmt |

### Training single-task / MTL source models

```bash
# From /home/ilias/TriLiteNet, inside docker:
docker run --rm --gpus '"device=GPU"' --shm-size=8g \
  -v /home/ilias/TriLiteNet:/workspace \
  -v /data/vgcmt/datasets/cityscapes:/data/vgcmt/datasets/cityscapes:ro \
  trilitenet:latest \
  python tools/train_cityscapes.py \
    --task mtl   # or: det | seg
    --size base --input-h 1024 --input-w 2048 \
    --epochs 300 --batch 16 --device 0
```

---

## §Results — Harvest & View

```bash
# From /home/ilias/AMROD
python3 tools/harvest_results.py            # update registry.json
python3 tools/harvest_results.py --print    # print all experiments
python3 tools/harvest_results.py --print --filter ACDC-10   # filter by benchmark
python3 tools/harvest_results.py --print --filter CS-C
python3 tools/harvest_results.py --print --filter SHIFT
```

Results are stored in `results/registry.json`.

### Key results summary (AP50 unless noted)

**Cityscapes → ACDC (short-task)**

| Method | AP50 | mIoU |
|:---|:---:|:---:|
| AMROD | 35.3% | — |
| CT-CMT-MTL v4b | — | — |
| TriLiteNet source | 4.4% | 27.4% |
| TriLiteNet CT-CMT | 1.1% | 14.1% |

**Cityscapes → ACDC (long-term, 10 rounds, mean AP50)**

| Method | R1 | R4 | R7 | R10 |
|:---|:---:|:---:|:---:|:---:|
| AMROD | 35.5 | 39.8 | 40.2 | 39.9 |
| CT-CMT-Det | 34.2 | 40.1 | 41.4 | **42.2** |
| CT-CMT-MTL v4b | 36.1 | 40.8 | 40.0 | 39.0 |

**Cityscapes → Cityscapes-C (short-task, mean AP50)**

| Method | Source | Mean AP50 |
|:---|:---:|:---:|
| AMROD (official) | MkRCNN | 19.0% |
| AMROD (our src) | MkRCNN | 17.2% |
| CT-CMT-Det (MkRCNN src) | MkRCNN | 17.0% |
| CT-CMT-MTL (PFN src) | PanFPN | 14.0% |

**Cityscapes → SHIFT (short-task, mean AP)**

| Method | AP | AP50 |
|:---|:---:|:---:|
| AMROD | 20.21 | ~37.6% |
| CT-CMT-Det | 18.27 | ~32.6% |
