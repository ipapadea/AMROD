#!/usr/bin/env bash
# Continual Cityscapes -> ACDC (fog -> night -> rain -> snow) with a CTTA
# meta-arch. State persists across weathers because detectron2's do_test
# iterates DATASETS.TEST while reusing the same model instance.
set -euo pipefail
GPU="$1"
TRACK="$2"        # amrod | cotta_semseg | ctcmt_mtl | ctcmt_det | ctcmt_seg
                  # ctcmt_mtl_no_ctcl | ctcmt_mtl_intra_ctcl
CFG=""
case "$TRACK" in
  amrod)
    CFG="detectron2/configs/Cityscapes/amrod_mask_rcnn_R_50_ACDC.yaml"
    ;;
  cotta_semseg)
    CFG="detectron2/configs/Cityscapes/cotta_semseg_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_panoptic_fpn_R_50_ACDC.yaml"
    ;;
  ctcmt_det)
    CFG="detectron2/configs/Cityscapes/ctcmt_det_only_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_seg)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_only_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_intra_ctcl)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_intra_ctcl_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v1)
    CFG="detectron2/configs/Cityscapes/ctcmt_v1_per_task_gate_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v2)
    CFG="detectron2/configs/Cityscapes/ctcmt_v2_cross_fisher_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v3)
    CFG="detectron2/configs/Cityscapes/ctcmt_v3_ctpv_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v4)
    CFG="detectron2/configs/Cityscapes/ctcmt_v4_proto_anchor_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v2v3)
    CFG="detectron2/configs/Cityscapes/ctcmt_v2v3_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v4b)
    CFG="detectron2/configs/Cityscapes/ctcmt_v4b_proto_pfn_R_50_ACDC.yaml"
    ;;
  amrod_v2)
    CFG="detectron2/configs/Cityscapes/amrod_v2_mask_rcnn_R_50_ACDC.yaml"
    ;;
  amrod_loopback)
    CFG="detectron2/configs/Cityscapes/amrod_loopback_R_50_ACDC.yaml"
    ;;
  cotta_semseg_loopback)
    CFG="detectron2/configs/Cityscapes/cotta_semseg_loopback_R_50_ACDC.yaml"
    ;;
  ctcmt_det_shift)
    CFG="detectron2/configs/Cityscapes/ctcmt_det_shift_R_50_CTTA.yaml"
    ;;
  source_only_shift)
    CFG="detectron2/configs/Cityscapes/source_only_shift.yaml"
    ;;
  amrod_shift)
    CFG="detectron2/configs/Cityscapes/amrod_shift_R_50_CTTA.yaml"
    ;;
  cotta_v2)
    CFG="detectron2/configs/Cityscapes/cotta_v2_semseg_R_50_ACDC.yaml"
    ;;
  tent)
    CFG="detectron2/configs/Cityscapes/tent_semseg_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_semfpn_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_aug)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_aug_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_no_proto)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_no_proto_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_aug_no_proto)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_aug_no_proto_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_high_conf)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_high_conf_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_aug_hc)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_aug_hc_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_aug_sp)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_aug_sp_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_small_aug)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_small_aug_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_sf_source_proto)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_sf_source_proto_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_proto_only)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_proto_only_R_50_ACDC.yaml"
    ;;
  ctcmt_seg_small_aug_no_v2)
    CFG="detectron2/configs/Cityscapes/ctcmt_seg_small_aug_no_v2_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl_v2)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_v2_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_ctcl_no_v2)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_ctcl_no_v2_pfn_R_50_ACDC.yaml"
    ;;
  source_only_foggy)
    CFG="detectron2/configs/Cityscapes/source_only_foggy_cityscapes.yaml"
    ;;
  amrod_foggy)
    CFG="detectron2/configs/Cityscapes/amrod_foggy_cityscapes.yaml"
    ;;
  amrod_v2_foggy)
    CFG="detectron2/configs/Cityscapes/amrod_v2_foggy_cityscapes.yaml"
    ;;
  ctcmt_v2_foggy_mtl)
    CFG="detectron2/configs/Cityscapes/ctcmt_v2_foggy_cityscapes_mtl.yaml"
    ;;
  ctcmt_mtl_ctcl_no_v2_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_ctcl_no_v2_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_ctcl_no_v2_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_ctcl_no_v2_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_ctcl_no_v2_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_ctcl_no_v2_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v2_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_v2_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v2_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_v2_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_v2_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_v2_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl_v2_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_v2_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl_v2_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_v2_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_mtl_no_ctcl_v2_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_v2_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_det_mr_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_det_mr_seed0_R_50_ACDC.yaml"
    ;;
  ctcmt_det_mr_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_det_mr_seed42_R_50_ACDC.yaml"
    ;;
  ctcmt_det_mr_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_det_mr_seed123_R_50_ACDC.yaml"
    ;;
  ctcmt_e1_ctpv_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e1_ctpv_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e1_ctpv_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e1_ctpv_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e1_ctpv_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e1_ctpv_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e2_entropy_ce_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e2_entropy_ce_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e2_entropy_ce_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e2_entropy_ce_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e2_entropy_ce_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e2_entropy_ce_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e3_teacher_aug_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e3_teacher_aug_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e3_teacher_aug_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e3_teacher_aug_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e3_teacher_aug_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e3_teacher_aug_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e4_dir_gate_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e4_dir_gate_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e4_dir_gate_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e4_dir_gate_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e4_dir_gate_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e4_dir_gate_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e5_adaptive_str_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e5_adaptive_str_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e5_adaptive_str_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e5_adaptive_str_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e5_adaptive_str_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e5_adaptive_str_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e6_all_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e6_all_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e6_all_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e6_all_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e6_all_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e6_all_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e7_e1_e3_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e7_e1_e3_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e7_e1_e3_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e7_e1_e3_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e7_e1_e3_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e7_e1_e3_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e4b_dir_gate_boost15_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_e4b_dir_gate_boost15_seed0_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e4b_dir_gate_boost15_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_e4b_dir_gate_boost15_seed42_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_e4b_dir_gate_boost15_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_e4b_dir_gate_boost15_seed123_pfn_R_50_ACDC.yaml"
    ;;
  ctcmt_loopback_v2_ctpv_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_loopback_v2_ctpv_seed0_R_50_ACDC.yaml"
    ;;
  ctcmt_loopback_v2_ctpv_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_loopback_v2_ctpv_seed42_R_50_ACDC.yaml"
    ;;
  ctcmt_loopback_v2_ctpv_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_loopback_v2_ctpv_seed123_R_50_ACDC.yaml"
    ;;
  ctcmt_loopback_det_mr_seed0)
    CFG="detectron2/configs/Cityscapes/ctcmt_loopback_det_mr_seed0_R_50_ACDC.yaml"
    ;;
  ctcmt_loopback_det_mr_seed42)
    CFG="detectron2/configs/Cityscapes/ctcmt_loopback_det_mr_seed42_R_50_ACDC.yaml"
    ;;
  ctcmt_loopback_det_mr_seed123)
    CFG="detectron2/configs/Cityscapes/ctcmt_loopback_det_mr_seed123_R_50_ACDC.yaml"
    ;;
  *) echo "unknown track $TRACK"; exit 1 ;;
esac

OUT_ROOT="/workspace/output/ctta_acdc/${TRACK}"

docker run --rm --gpus "\"device=${GPU}\"" --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -v /home/ilias/AMROD:/workspace/amrod \
  -v /data/vgcmt/datasets/cityscapes:/data/vgcmt/datasets/cityscapes:ro \
  -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro \
  -v /data/ilias/panoptic_fpn/output/coco_annotations:/datasets/annotations:ro \
  -v /data/ilias/acdc:/datasets/ACDC:ro \
  -v /data/vgcmt/datasets/cityscapes_foggy:/datasets/cityscapes_foggy:ro \
  -v /data/ilias/shift_amrod:/datasets/shift:ro \
  -v /data/ilias/shift/discrete/images:/data/ilias/shift/discrete/images:ro \
  -v "/data/vgcmt/downloads/amrod/extracted/model weight":/workspace/weights:ro \
  -v /data/ilias/panoptic_fpn/output:/workspace/output \
  -w /workspace/amrod \
  -e DETECTRON2_DATASETS=/datasets \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  amrod:latest bash -c "
    export HOME=/tmp
    pip install --quiet --user --no-warn-script-location shapely 2>&1 | tail -1
    echo '>>> CTTA ${TRACK} continual on acdc_fog->night->rain->snow <<<'
    python detectron2/tools/train_net.py \
      --config-file ${CFG} \
      --eval-only --num-gpus 1 \
      OUTPUT_DIR ${OUT_ROOT} 2>&1
    echo CTTA_${TRACK}_DONE
  "
