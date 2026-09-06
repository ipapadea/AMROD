# RUNBOOK — Decisions 1B (D2 pipeline) + 2b (AMROD-pinned corruption regen)

## Goal

Fair D2-vs-mmdet cross-framework comparison on the SAME Cityscapes-C images:

```
                   AMROD ckpt          Ours COCO ckpt
   D2 pipeline     [need to run]       [need to run]
   mmdet pipeline  [DONE 13.12]        [DONE 23.18]      (mean mAP@50 on AMROD-corr set)
```

The paper's Table 2 "Source" row is 15.1 mAP@50. If our port is faithful, the
top-left cell (D2 + AMROD-ckpt + AMROD-corr) should land at ~15.1.

## Current state (verified 2026-07-27)

- ✅ Docker image `amrod:latest` built (15 GB, 2026-07-24)
- ✅ AMROD-pinned corruption set: `/data/vgcmt/datasets/cityscapes_c_amrod/`  (all 12 corr × 500 imgs)
- ✅ AMROD D2 ckpt (symlink): `/data/vgcmt/models/amrod_d2/cityscapes_train_final.pth`
- ✅ Our mmdet→D2 ckpt: `/data/vgcmt/work_dirs/ours_d2/ours_cityscapes_d2.pth`
- ✅ mmdet-pipeline evals on BOTH corruption sets × BOTH ckpts (4 summary.json)
- ❌ D2-pipeline evals — need to run (blocked last time by dead symlink)

Fixed for this run: `docker-compose.yml` now mounts `/data/vgcmt/downloads:ro` so
the AMROD ckpt symlink resolves. `scripts/run_all_in_tmux.sh` also has a
belt-and-suspenders fallback that materializes a real copy if the symlink dies.

## The single command

```bash
bash /home/ilias/AMROD/scripts/run_all_in_tmux.sh
```

That fires everything as **four independent tmux sessions** and returns
immediately. You can safely close the shell after — everything keeps running.

Attach to whichever you want:

```bash
tmux ls                                # list all sessions
tmux attach -t amrod-eval-amrod        # GPU 0, D2 + AMROD ckpt
tmux attach -t amrod-eval-ours         # GPU 1, D2 + ours ckpt
tmux attach -t amrod-aggregate         # waits + prints cross-tab
tmux attach -t amrod-monitor           # nvidia-smi + status watch
# detach any of them with:  Ctrl-b d
```

**Wall time**: ~6–8 minutes (12 corruptions × ~30s per GPU, parallel on 2 GPUs).

**Final output**: `/tmp/cross_tab_final.txt` — 2×2 comparison table plus
paper-15.1 sanity check.

**Marker files** (appear as steps complete):
```
/data/vgcmt/status/d2_amrod.done
/data/vgcmt/status/d2_ours.done
/data/vgcmt/status/all.done
```

## Just want the final table right now (partial results ok)?

```bash
python3 /home/ilias/AMROD/tools/cross_tabulate.py
```

That reads whichever summary.json files exist and prints a table with `--` for
missing cells.

## If something breaks

### The docker image is missing
Rebuild:
```bash
cd /home/ilias/AMROD && HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build
```
Takes ~5 minutes (torch cached).

### The AMROD-pinned corruption data was deleted
Regenerate (idempotent, resumable):
```bash
bash /home/ilias/AMROD/scripts/regen_cityscapes_c.sh
```
Wall time ~15 minutes on 24 CPU workers (glass_blur/zoom_blur/elastic_transform dominate).

### The `ours_cityscapes_d2.pth` converted ckpt was deleted
Re-run the converter (inside the vgcmt container):
```bash
cd /home/ilias/vgcmt
docker compose run --rm vgcmt bash -lc "\
    mkdir -p /workspace/vgcmt/work_dirs/ours_d2 && \
    python tools/convert_mmdet_to_d2.py \
        /workspace/vgcmt/work_dirs/source_frcnn_r50_cityscapes_coco/best_coco_bbox_mAP.pth \
        /workspace/vgcmt/work_dirs/ours_d2/ours_cityscapes_d2.pth"
```
Takes ~30 seconds.

### The AMROD D2 ckpt symlink is dead
Materialize a real copy:
```bash
cp "/data/vgcmt/downloads/amrod/extracted/model weight/cityscapes_train_final.pth" \
   /data/vgcmt/models/amrod_d2/cityscapes_train_final.pth
```

### Want to see the mmdet-side results that already exist?
```bash
python3 -c "
import json
for f in [
    '/data/vgcmt/work_dirs/cityscapes_c_source_only_amrod/summary.json',
    '/data/vgcmt/work_dirs/cityscapes_c_source_only_ours_coco/summary.json',
    '/data/vgcmt/work_dirs/cityscapes_c_source_only_amrod_amrodcorr/summary.json',
    '/data/vgcmt/work_dirs/cityscapes_c_source_only_ours_coco_amrodcorr/summary.json',
]:
    s = json.load(open(f))
    print(f'{f.split(chr(47))[-2]:<45}  mean_mAP_50 = {s[\"mean_mAP_50\"]*100:.2f}')
"
```

## File inventory (all files needed for this pipeline)

**AMROD repo** (`/home/ilias/AMROD/`):
- `Dockerfile` — image recipe (deps pinned to AMROD's requirements)
- `docker-compose.yml` — mounts /data/vgcmt/{datasets,models,work_dirs,downloads}
- `tools/corrgen_amrod.py` — Decision 2b corruption regen (already run)
- `tools/d2_source_only_eval.py` — plain GeneralizedRCNN eval on one corruption
- `tools/aggregate_d2_cityscapes_c.py` — per-run summary.json
- `tools/cross_tabulate.py` — 2×2 cross-framework table
- `scripts/regen_cityscapes_c.sh` — Decision 2b driver (already run)
- `scripts/eval_d2_cityscapes_c.sh` — 12-corr loop for one ckpt
- `scripts/run_all_in_tmux.sh` — MAIN ENTRY POINT

**vgcmt repo** (`/home/ilias/vgcmt/`):
- `tools/convert_mmdet_to_d2.py` — mmdet → D2 ckpt (round-trip bit-exact vs
  `convert_d2_to_mmdet.py`; already used to produce `ours_cityscapes_d2.pth`)
- `tools/convert_d2_to_mmdet.py` — D2 → mmdet ckpt (already present)
- `configs/continuous/mean_teacher_frcnn/cityscapes_c_source_only.py` — reads
  `CITYSCAPES_C_CORRUPTION` and `CITYSCAPES_C_SET` env vars
- `scripts/adapt.sh` — forwards both env vars into the vgcmt container
- `scripts/eval_cityscapes_c.sh` — 12-corr loop for mmdet (accepts corr-set arg)
