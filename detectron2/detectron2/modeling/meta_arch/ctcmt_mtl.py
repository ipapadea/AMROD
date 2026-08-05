"""CT-CMT-MTL adapter for detectron2 PanopticFPN.

Extends the published CT-CMT idea (Moraiti et al. 2026, single-task detection
via YOLOX) to multi-task learning on Panoptic FPN (det + sem-seg).

Three loss terms during adaptation, all computed from the STUDENT model and
supervised by the frozen TEACHER (EMA of student):

  1. Detection consistency: teacher inference -> boxes + classes filtered by
     dynamic per-class thresholds (Wei et al. 2024 style, matching AMROD).
     Student is trained with these as gt_instances via the standard Faster
     R-CNN proposal + roi head losses. Mask head is disabled during adaptation
     (no reliable pseudo-masks).

  2. Semantic segmentation consistency: soft cross-entropy of student
     sem_seg logits against teacher sem_seg softmax (matches CoTTA).

  3. Cross-task supervised contrastive (CT-CL, paper's core contribution):
     For each pseudo-detected box b with class c, build TWO feature views
     from the student FPN feature maps:

         z_det(b) = RoI-Align at b                             # det view
         z_seg(b) = mask-weighted mean inside b, weighted by
                     teacher_seg_probs[..., c][b]              # seg view

     Same-class pairs (det<->det, seg<->seg, det<->seg) are pulled together;
     different-class pairs pushed apart. Creates a shared cross-task
     representation.

Teacher weights are updated as EMA of student. All trainable weight+bias
params in the student are stochastically restored to source with prob
``COTTA_RESTORE_PROB`` per step (CoTTA convention).

Reference:
    Moraiti et al. 2026 (EJAI) --- single-task CT-CMT for YOLOX.
    Wang et al. 2022 (CVPR) --- CoTTA (segmentation CTTA).
    Wei et al. 2024 --- AMROD (dynamic per-class thresholds).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.solver import build_optimizer
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage

from ..postprocessing import detector_postprocess, sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["CTCMT_MTL"]

# Cityscapes taxonomy: detection classes 0..7 (thing_classes) map onto
# semantic-segmentation trainIds 11..18 (person, rider, car, truck, bus,
# train, motorcycle, bicycle). Used for the CT-CL seg view when computing
# mask-weighted features per bbox class.
_DET_TO_SEG_CLASS_CITYSCAPES = (11, 12, 13, 14, 15, 16, 17, 18)


# =====================================================================
# Supervised contrastive loss (Khosla et al. 2020) --- compact inline.
# =====================================================================
def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """features: (N, D) L2-normalized. labels: (N,) int."""
    N = features.size(0)
    if N < 2:
        return features.new_zeros(())
    device = features.device
    # Similarity matrix.
    logits = torch.matmul(features, features.t()) / temperature
    # For numerical stability.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    # Positive mask (same class), excluding self.
    labels = labels.view(-1, 1)
    mask_pos = torch.eq(labels, labels.t()).float().to(device)
    diag = torch.eye(N, device=device)
    mask_pos = mask_pos - diag
    exp_logits = torch.exp(logits) * (1.0 - diag)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    n_pos = mask_pos.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return features.new_zeros(())
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


# =====================================================================
# BN convention (TENT / CoTTA)
# =====================================================================
def _configure_batch_stats_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


# =====================================================================
# Dynamic threshold (Wei et al. 2024 style, same math as AMROD)
# =====================================================================
def _dyn_thresholds(prev, per_class_mean_scores, alpha: float, gamma: float,
                    lo: float, hi: float):
    new = []
    for th, mean in zip(prev, per_class_mean_scores):
        if mean > 0:
            th = gamma * th + (1.0 - gamma) * alpha * math.sqrt(mean)
        th = max(min(th, hi), lo)
        new.append(th)
    return new


# =====================================================================
# CT-CMT-MTL meta-arch
# =====================================================================
@META_ARCH_REGISTRY.register()
class CTCMT_MTL(nn.Module):
    """Cross-Task Consistent Mean Teacher for detectron2 PanopticFPN."""

    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        anchor: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.anchor = anchor
        self.optimizer = optimizer

        self.num_classes = int(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        # MR-CNN configs have no SEM_SEG_HEAD; keep num_seg_classes optional.
        _seg_head_cfg = getattr(cfg.MODEL, "SEM_SEG_HEAD", None)
        self.num_seg_classes = int(_seg_head_cfg.NUM_CLASSES) if _seg_head_cfg is not None else 0

        # Mean-teacher det hyperparams (mirrors AMROD).
        self.threshold_init = float(cfg.SOLVER.THRESHOLD_INIT)
        self.thresholds_max = float(cfg.SOLVER.THRESHOLD_MAX)
        self.thresholds_mini = float(cfg.SOLVER.THRESHOLD_MINI)
        self.alpha_dt = float(cfg.SOLVER.ALPHA_DT)
        self.gamma_dt = float(cfg.SOLVER.GAMMA_DT)
        self.thresholds = [self.threshold_init] * self.num_classes

        # Score-EM gating (skip step when teacher confidence is stable).
        self.score_em = float(cfg.SOLVER.SCORE_EM)
        self.score_gamma = float(cfg.SOLVER.SCORE_GAMMA)
        self.score_thresh = float(cfg.SOLVER.SCORE_THRESH)
        # When set, ignore the gate and let every image contribute a step.
        self.skip_score_em_gate = bool(getattr(cfg.SOLVER, "CTCMT_SKIP_SCORE_EM_GATE", False))

        # Loss weights.
        self.weight_det = float(cfg.SOLVER.CTCMT_WEIGHT_DET)
        self.weight_seg = float(cfg.SOLVER.CTCMT_WEIGHT_SEG)
        self.weight_ctcl = float(cfg.SOLVER.CTCMT_WEIGHT_CTCL)
        # Cross-task consistency regularizer: pixels inside a teacher-detected
        # bbox should be classified by the student seg head as that box's class.
        self.weight_ctcr = float(getattr(cfg.SOLVER, "CTCMT_WEIGHT_CTCR", 0.0))
        # Single-task ablation switches: disable one task branch entirely so
        # this meta-arch degenerates to a fair single-task-on-MTL-source baseline.
        self.det_only = bool(getattr(cfg.SOLVER, "CTCMT_DET_ONLY", False))
        self.seg_only = bool(getattr(cfg.SOLVER, "CTCMT_SEG_ONLY", False))
        # CoTTA-style multi-scale augmentation-averaged seg pseudo-labels.
        self.seg_aug_enabled = bool(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_ENABLED", False))
        self.seg_aug_conf_thresh = float(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_CONF_THRESH", 0.9))
        self.seg_aug_scales = tuple(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_SCALES", (1.0,)))
        self.seg_aug_flips = tuple(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_FLIPS", (False,)))

        # Cross-task contrastive.
        self.ctcl_enabled = bool(cfg.SOLVER.CTCMT_CTCL_ENABLED)
        self.ctcl_include_seg_view = bool(cfg.SOLVER.CTCMT_CTCL_SEG_VIEW)
        self.ctcl_temperature = float(cfg.SOLVER.CTCMT_CTCL_TEMPERATURE)
        self.ctcl_roi_output = tuple(cfg.SOLVER.CTCMT_CTCL_ROI_OUTPUT)
        self.ctcl_proj_dim = int(cfg.SOLVER.CTCMT_CTCL_PROJ_DIM)

        # CoTTA-style EMA + stochastic restore.
        self.ema_decay = float(cfg.SOLVER.MT)
        self.restore_prob = float(cfg.SOLVER.RST_M)

        # --- Novel extension V1: per-task decoupled adaptation gates ---
        # Instead of one global gate, each task branch fires independently when
        # its own confidence is low (= domain shift detected for that task).
        self.per_task_gate = bool(getattr(cfg.SOLVER, "CTCMT_PER_TASK_GATE", False))
        self.per_task_gate_det_thresh = float(getattr(cfg.SOLVER, "CTCMT_PER_TASK_GATE_DET_THRESH", 0.8))
        self.per_task_gate_seg_thresh = float(getattr(cfg.SOLVER, "CTCMT_PER_TASK_GATE_SEG_THRESH", 0.8))

        # --- Novel extension V2: task-aware stochastic restore ---
        # Shared backbone / FPN params are used by both tasks; restore them
        # with a lower probability than task-specific head params.
        self.cross_task_fisher = bool(getattr(cfg.SOLVER, "CTCMT_CROSS_TASK_FISHER", False))
        # backbone restore rate = restore_prob * this factor (< 1 = more protection)
        self.backbone_rst_factor = float(getattr(cfg.SOLVER, "CTCMT_BACKBONE_RST_FACTOR", 0.1))

        # --- Novel extension V3: cross-task pseudo-label verification ---
        # Reject a det box as pseudo-label if the seg head disagrees with its class.
        self.ctpv_enabled = bool(getattr(cfg.SOLVER, "CTCMT_CTPV_ENABLED", False))
        self.ctpv_thresh = float(getattr(cfg.SOLVER, "CTCMT_CTPV_THRESH", 0.3))

        # --- Novel extension V4: cross-task prototype anchor ---
        # Running EMA prototypes (det-view + seg-view) per class. Updated only
        # when both views agree. Add a weak pull toward stored prototypes.
        self.proto_anchor = bool(getattr(cfg.SOLVER, "CTCMT_PROTO_ANCHOR", False))
        self.proto_ema = float(getattr(cfg.SOLVER, "CTCMT_PROTO_EMA", 0.999))
        self.proto_weight = float(getattr(cfg.SOLVER, "CTCMT_PROTO_WEIGHT", 0.01))
        self._det_protos: Dict[int, torch.Tensor] = {}   # class -> (D,) det-view
        self._seg_protos: Dict[int, torch.Tensor] = {}   # class -> (D,) seg-view

        # --- Enhancement E2: entropy-weighted soft-CE (down-weight uncertain pixels) ---
        self.entropy_weighted_ce = bool(getattr(cfg.SOLVER, "CTCMT_ENTROPY_WEIGHTED_CE", False))

        # --- Enhancement E3: trigger seg aug-avg on TEACHER entropy, not anchor confidence ---
        self.aug_trigger_teacher_entropy = bool(getattr(cfg.SOLVER, "CTCMT_AUG_TRIGGER_TEACHER_ENTROPY", False))
        self.aug_teacher_entropy_thresh = float(getattr(cfg.SOLVER, "CTCMT_AUG_TEACHER_ENTROPY_THRESH", 0.3))

        # --- Enhancement E4: directional score-EM gate (skip on stable, adapt on shift) ---
        self.directional_gate = bool(getattr(cfg.SOLVER, "CTCMT_DIRECTIONAL_GATE", False))
        self.dir_gate_stable_band = float(getattr(cfg.SOLVER, "CTCMT_DIR_GATE_STABLE_BAND", 0.4))
        self.dir_gate_boost = float(getattr(cfg.SOLVER, "CTCMT_DIR_GATE_BOOST", 2.0))
        # Boost applied to CT-CL/CT-CR when a downward shift is detected.
        self._cl_boost = 1.0

        # --- Enhancement E5: adaptive STR (η depends on measured shared-trunk drift) ---
        self.adaptive_str = bool(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR", False))
        self.adaptive_str_base = float(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR_BASE", 0.1))
        self.adaptive_str_boost = float(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR_BOOST", 0.4))
        self.adaptive_str_pivot = float(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR_PIVOT", 0.05))
        self._backbone_source_norm = None   # cached lazily after _source_params snapshot

        self.iter = 0

        # Snapshot every trainable weight/bias in the anchor for stochastic restore.
        self._source_params: Dict[str, torch.Tensor] = {}
        for nm, m in self.anchor.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name in ("weight", "bias"):
                    self._source_params[f"{nm}.{np_name}"] = p.detach().clone()

        # Contrastive projection heads (built lazily on first CTCL call once
        # we know the FPN channel count).
        self._proj_det = None
        self._proj_seg = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        # Student meta-arch is configurable so CTCMT_MTL can wrap either a
        # PanopticFPN (MTL / seg-only / det-only on PFN source) or a
        # GeneralizedRCNN (det-only on a Mask R-CNN source).
        student_arch_name = getattr(cfg.MODEL, "CTCMT_STUDENT_META_ARCH", "PanopticFPN")
        pfn_cls = META_ARCH_REGISTRY.get(student_arch_name)
        weights_path = cfg.MODEL.WEIGHTS

        def _build_and_load(train_mode: bool, freeze: bool, disable_mask_head: bool = True):
            m = pfn_cls(cfg)
            # Use .load() (not resume_or_load) so FrozenBN buffers reliably
            # come from the checkpoint.
            DetectionCheckpointer(m).load(weights_path)
            m.to(torch.device(cfg.MODEL.DEVICE))
            if train_mode:
                m.train()
            else:
                m.eval()
            if freeze:
                for p in m.parameters():
                    p.requires_grad_(False)
            # No pseudo-masks -> disable mask head on student/anchor to skip
            # wasteful mask computation during adaptation. Keep it on the
            # teacher so the evaluator receives pred_masks (e.g. cityscapes).
            if disable_mask_head and hasattr(m.roi_heads, "mask_on"):
                m.roi_heads.mask_on = False
            return m

        student = _build_and_load(train_mode=True,  freeze=False, disable_mask_head=True)
        teacher = _build_and_load(train_mode=True,  freeze=True,  disable_mask_head=False)
        anchor  = _build_and_load(train_mode=False, freeze=True,  disable_mask_head=True)

        optimizer = build_optimizer(cfg, student)
        return {
            "student": student,
            "teacher": teacher,
            "anchor": anchor,
            "optimizer": optimizer,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.student.pixel_mean.device

    # ------------------------------------------------------------------
    # Teacher pseudo-labels via dynamic per-class thresholds + score-EM gate.
    # Returns (list[Instances], keep_step: bool, score_summary: str)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _teacher_pseudo(self, batched_inputs):
        # Raw teacher outputs (pre-postprocess), in the RESIZED image frame.
        # PanopticFPN returns (det, sem); GeneralizedRCNN returns det only.
        teacher_out = self.teacher.inference(batched_inputs, do_postprocess=False)
        if isinstance(teacher_out, tuple) and len(teacher_out) == 2:
            detector_results, sem_seg_results = teacher_out
        else:
            detector_results, sem_seg_results = teacher_out, None
        inst = detector_results[0]
        if len(inst) == 0:
            return detector_results, sem_seg_results, False

        # Score-EM gate: skip step if teacher confidence is stable.
        valid_mask = inst.scores > 0.1
        if not valid_mask.any():
            return detector_results, sem_seg_results, False
        mean_all = float(inst.scores[valid_mask].mean().cpu())
        keep_step = True
        if self.score_em > 0 and not self.skip_score_em_gate:
            ratio = mean_all / self.score_em
            if self.directional_gate:
                # E4: skip only when high AND stable; force adapt (and boost CT-CL) on shift down.
                if mean_all >= self.score_em and abs(ratio - 1.0) < self.dir_gate_stable_band:
                    keep_step = False   # already adapted
                    self._cl_boost = 1.0
                elif ratio < 0.6:
                    self._cl_boost = self.dir_gate_boost   # sharp drop = shift onset
                else:
                    self._cl_boost = 1.0
            else:
                if ratio > self.score_thresh or (1.0 / max(ratio, 1e-6)) > self.score_thresh:
                    keep_step = False
        self.score_em = self.score_gamma * self.score_em + (1.0 - self.score_gamma) * mean_all

        # Per-class mean score -> new dynamic thresholds.
        per_class_mean = [0.0] * self.num_classes
        classes = inst.pred_classes[valid_mask]
        scores = inst.scores[valid_mask]
        for c in range(self.num_classes):
            idx = classes == c
            if int(idx.sum()) > 0:
                per_class_mean[c] = float(scores[idx].mean())
        self.thresholds = _dyn_thresholds(
            self.thresholds, per_class_mean,
            self.alpha_dt, self.gamma_dt,
            self.thresholds_mini, self.thresholds_max,
        )

        # Filter with dyn thresholds.
        thr = torch.tensor(self.thresholds, device=inst.scores.device)
        keep = inst.scores >= thr[inst.pred_classes.long()]
        filtered = Instances(inst.image_size)
        filtered.pred_boxes = Boxes(inst.pred_boxes.tensor[keep])
        filtered.pred_classes = inst.pred_classes[keep]
        filtered.scores = inst.scores[keep]
        for k in inst.get_fields():
            if k not in ("pred_boxes", "pred_classes", "scores"):
                filtered.set(k, inst.get(k)[keep])
        return [filtered], sem_seg_results, keep_step

    # ------------------------------------------------------------------
    # Convert filtered teacher instances -> pseudo gt_instances the student
    # detection heads expect (uses gt_boxes / gt_classes).
    # ------------------------------------------------------------------
    @staticmethod
    def _to_gt_instances(instances_list):
        out = []
        for inst in instances_list:
            g = Instances(inst.image_size)
            g.gt_boxes = Boxes(inst.pred_boxes.tensor.clone())
            g.gt_classes = inst.pred_classes.long().clone()
            out.append(g)
        return out

    # ------------------------------------------------------------------
    # Cross-task contrastive on student FPN features.
    # ------------------------------------------------------------------
    def _ctcl_loss(self, student_features, pseudo_instances, teacher_sem_probs):
        inst = pseudo_instances[0]
        if len(inst) == 0:
            return student_features[next(iter(student_features))].new_zeros(())

        # Take the deepest FPN level that's in ROI box features for a compact view.
        feat_key = self.student.roi_heads.box_in_features[-1]
        feat = student_features[feat_key]  # (B, C, H, W)
        C = feat.size(1)

        # Convert boxes from RESIZED input coords to feat coords via FPN stride.
        stride = self.student.backbone.output_shape()[feat_key].stride
        boxes = inst.pred_boxes.tensor / float(stride)
        classes = inst.pred_classes.long()

        # ---- Det view: RoI-align pooled features -> mean over spatial -> L2 norm.
        # Raw features (no random-init projection heads) match the shift-tta reference.
        rois = torch.cat(
            [torch.zeros((boxes.size(0), 1), device=boxes.device), boxes], dim=1
        )
        pooled = roi_align(feat, rois, output_size=self.ctcl_roi_output,
                           spatial_scale=1.0, aligned=True)  # (N, C, h, w)
        z_det = pooled.mean(dim=(2, 3))                         # (N, C)
        z_det = F.normalize(z_det, dim=1)

        views = [z_det]
        labels = [classes]

        if self.ctcl_include_seg_view and teacher_sem_probs is not None:
            # Downsample teacher probs to feat resolution.
            probs_feat = F.interpolate(
                teacher_sem_probs, size=feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            H, W = feat.shape[-2:]
            z_seg_list = []
            for b, c in zip(boxes.detach().cpu().tolist(), classes.detach().cpu().tolist()):
                # Map detection class -> seg channel (Cityscapes 8->19 taxonomy).
                if 0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES):
                    seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
                else:
                    continue
                if seg_c >= self.num_seg_classes:
                    continue
                x1, y1, x2, y2 = [max(int(round(v)), 0) for v in b]
                x2 = min(x2, W)
                y2 = min(y2, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_f = feat[0:1, :, y1:y2, x1:x2]                     # (1, C, h, w)
                crop_p = probs_feat[0:1, seg_c:seg_c + 1, y1:y2, x1:x2] # (1, 1, h, w)
                w_sum = crop_p.sum().clamp(min=1e-6)
                pooled_seg = (crop_f * crop_p).sum(dim=(2, 3)) / w_sum  # (1, C)
                # Label with the DETECTION class so SupCon pulls det-view and
                # seg-view of the same object together (cross-task pairs).
                z_seg_list.append((pooled_seg.squeeze(0), c))
            if z_seg_list:
                z_seg = torch.stack([z for z, _ in z_seg_list], dim=0)
                z_seg = F.normalize(z_seg, dim=1)
                seg_classes = torch.tensor(
                    [c for _, c in z_seg_list], device=z_seg.device, dtype=classes.dtype
                )
                views.append(z_seg)
                labels.append(seg_classes)

        z = torch.cat(views, dim=0)
        y = torch.cat(labels, dim=0)
        return _supcon_loss(z, y, temperature=self.ctcl_temperature)

    # ------------------------------------------------------------------
    # Cross-task consistency regularizer: student seg logits inside teacher
    # boxes must classify as that box's (seg-taxonomy) class.
    # ------------------------------------------------------------------
    def _ctcr_loss(self, s_seg_logits, pseudo_instances):
        inst = pseudo_instances[0]
        if len(inst) == 0:
            return None
        B, K, H, W = s_seg_logits.shape
        target = torch.full((B, H, W), 255, dtype=torch.long, device=s_seg_logits.device)
        # Boxes are in the RESIZED input frame. Rescale to seg-logit grid.
        img_h, img_w = inst.image_size
        sx = W / max(img_w, 1)
        sy = H / max(img_h, 1)
        boxes = inst.pred_boxes.tensor.detach()
        classes = inst.pred_classes.detach().long().tolist()
        for j, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
            c = classes[j]
            if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                continue
            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            if seg_c >= K:
                continue
            x1i = max(int(round(x1 * sx)), 0); y1i = max(int(round(y1 * sy)), 0)
            x2i = min(int(round(x2 * sx)), W); y2i = min(int(round(y2 * sy)), H)
            if x2i <= x1i or y2i <= y1i:
                continue
            target[0, y1i:y2i, x1i:x2i] = seg_c
        if (target != 255).sum() == 0:
            return None
        return F.cross_entropy(s_seg_logits, target, ignore_index=255)

    # ------------------------------------------------------------------
    # EMA + stochastic restore.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_teacher(self):
        d = self.ema_decay
        s_state = self.student.state_dict()
        for k, v in self.teacher.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s_state[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(s_state[k])

    @torch.no_grad()
    def _stochastic_restore(self):
        if self.restore_prob <= 0.0:
            return
        _shared_prefixes = ("backbone.", "fpn.", "proposal_generator.anchor_generator.")
        # E5: measure current shared-trunk drift and adapt the backbone-restore factor.
        eff_backbone_factor = self.backbone_rst_factor
        if self.adaptive_str and self.cross_task_fisher:
            if self._backbone_source_norm is None:
                self._backbone_source_norm = 0.0
                for key, src in self._source_params.items():
                    if any(key.startswith(pfx) for pfx in _shared_prefixes):
                        self._backbone_source_norm += float(src.pow(2).sum().cpu())
                self._backbone_source_norm = math.sqrt(self._backbone_source_norm) + 1e-6
            drift_sq = 0.0
            student_state = self.student.state_dict()
            for key, src in self._source_params.items():
                if any(key.startswith(pfx) for pfx in _shared_prefixes):
                    if key in student_state:
                        s = student_state[key]
                        drift_sq += float((s - src.to(s.device)).pow(2).sum().cpu())
            drift = math.sqrt(drift_sq) / self._backbone_source_norm
            # Sigmoid boost: base + boost * σ(10*(drift-pivot))
            eff_backbone_factor = self.adaptive_str_base + self.adaptive_str_boost * (
                1.0 / (1.0 + math.exp(-10.0 * (drift - self.adaptive_str_pivot)))
            )
        for nm, m in self.student.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name not in ("weight", "bias") or not p.requires_grad:
                    continue
                key = f"{nm}.{np_name}"
                src = self._source_params.get(key)
                if src is None:
                    continue
                rst = self.restore_prob
                if self.cross_task_fisher and any(key.startswith(pfx) for pfx in _shared_prefixes):
                    rst = rst * eff_backbone_factor
                if rst <= 0.0:
                    continue
                mask = (torch.rand_like(p) < rst).float()
                src_dev = src.to(p.device, non_blocking=True)
                p.data.mul_(1.0 - mask).add_(src_dev * mask)

    # V3: reject a pseudo-box if the seg head disagrees with its class.
    @torch.no_grad()
    def _ctpv_filter(self, instances, teacher_seg_probs):
        """Cross-task pseudo-label verification: keep box only when the seg
        head assigns ≥ ctpv_thresh fraction of its pixels to the matching
        seg class.  Returns filtered Instances."""
        inst = instances[0]
        if len(inst) == 0 or teacher_seg_probs is None:
            return instances
        _, K, H, W = teacher_seg_probs.shape
        img_h, img_w = inst.image_size
        sx = W / max(img_w, 1); sy = H / max(img_h, 1)
        keep = []
        for j, (box, c) in enumerate(zip(inst.pred_boxes.tensor.tolist(),
                                          inst.pred_classes.tolist())):
            if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                keep.append(True); continue
            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            if seg_c >= K:
                keep.append(True); continue
            x1, y1, x2, y2 = box
            x1i = max(int(round(x1 * sx)), 0); y1i = max(int(round(y1 * sy)), 0)
            x2i = min(int(round(x2 * sx)), W); y2i = min(int(round(y2 * sy)), H)
            if x2i <= x1i or y2i <= y1i:
                keep.append(False); continue
            region = teacher_seg_probs[0, :, y1i:y2i, x1i:x2i]  # (K, h, w)
            pred_class = region.argmax(dim=0)  # (h, w)
            agreement = float((pred_class == seg_c).float().mean())
            keep.append(agreement >= self.ctpv_thresh)
        if all(keep):
            return instances
        keep_t = torch.tensor(keep, device=inst.pred_boxes.tensor.device)
        filtered = Instances(inst.image_size)
        filtered.pred_boxes = Boxes(inst.pred_boxes.tensor[keep_t])
        filtered.pred_classes = inst.pred_classes[keep_t]
        filtered.scores = inst.scores[keep_t]
        return [filtered]

    # V4: update cross-task prototype anchors; returns prototype pull loss.
    def _proto_anchor_update_and_loss(self, features, pseudo_inst, teacher_seg_probs):
        inst = pseudo_inst[0]
        if len(inst) == 0:
            return features[next(iter(features))].new_zeros(())
        feat_key = self.student.roi_heads.box_in_features[-1]
        feat = features[feat_key]  # (B, C, H, W)
        stride = self.student.backbone.output_shape()[feat_key].stride
        boxes = (inst.pred_boxes.tensor / float(stride)).detach()
        classes = inst.pred_classes.detach().long().tolist()
        if teacher_seg_probs is not None:
            probs_feat = F.interpolate(teacher_seg_probs, size=feat.shape[-2:],
                                        mode="bilinear", align_corners=False)
        else:
            probs_feat = None
        H, W = feat.shape[-2:]
        proto_loss = feat.new_zeros(())
        for b, c in zip(boxes.tolist(), classes):
            if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                continue
            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            x1, y1, x2, y2 = [max(int(round(v)), 0) for v in b]
            x2 = min(x2, W); y2 = min(y2, H)
            if x2 <= x1 or y2 <= y1:
                continue
            # Det view.
            z_det = feat[0:1, :, y1:y2, x1:x2].mean(dim=(2, 3)).squeeze(0)
            z_det_n = F.normalize(z_det, dim=0)
            # Seg view (mask-weighted).
            if probs_feat is not None and seg_c < probs_feat.shape[1]:
                w = probs_feat[0, seg_c, y1:y2, x1:x2].clamp(min=1e-6)
                w = w / w.sum()
                z_seg = (feat[0, :, y1:y2, x1:x2] * w.unsqueeze(0)).sum(dim=(1, 2))
            else:
                z_seg = z_det.clone()
            z_seg_n = F.normalize(z_seg, dim=0)
            # Only update prototype if both views roughly agree (cosine sim > 0).
            cross_sim = float((z_det_n * z_seg_n).sum().item())
            with torch.no_grad():
                if cross_sim > 0.0:
                    z_det_np = z_det_n.detach()
                    z_seg_np = z_seg_n.detach()
                    if c not in self._det_protos:
                        self._det_protos[c] = z_det_np.clone()
                        self._seg_protos[c] = z_seg_np.clone()
                    else:
                        a = self.proto_ema
                        self._det_protos[c] = a * self._det_protos[c] + (1 - a) * z_det_np
                        self._seg_protos[c] = a * self._seg_protos[c] + (1 - a) * z_seg_np
            # Pull current features toward stored prototypes.
            if c in self._det_protos:
                proto_det = self._det_protos[c].to(feat.device)
                proto_seg = self._seg_protos[c].to(feat.device)
                proto_loss = proto_loss + (1.0 - (z_det_n * proto_det.detach()).sum())
                proto_loss = proto_loss + (1.0 - (z_seg_n * proto_seg.detach()).sum())
        return proto_loss

    # ------------------------------------------------------------------
    # Forward = one CTTA step. Returns per-image dict with 'sem_seg' and
    # 'instances' (no panoptic combine — avoids pred_masks dependency).
    # ------------------------------------------------------------------
    # CoTTA-style multi-scale aug-averaged teacher seg probs.
    # Only invoked when seg_aug_enabled AND anchor confidence is below the
    # per-image threshold, keeping runtime low on easy inputs.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _teacher_aug_avg_seg_probs(self, image_tensor):
        H, W = image_tensor.shape[-2:]
        divisor = int(getattr(self.teacher.backbone, "size_divisibility", 32) or 32)
        def _snap(v):
            v = max(int(round(v)), divisor)
            return ((v + divisor - 1) // divisor) * divisor
        accum = None
        n = 0
        for s in self.seg_aug_scales:
            Hs, Ws = _snap(H * s), _snap(W * s)
            for flip in self.seg_aug_flips:
                x_aug = F.interpolate(image_tensor, size=(Hs, Ws),
                                      mode="bilinear", align_corners=False)
                if flip:
                    x_aug = torch.flip(x_aug, dims=[-1])
                feats = self.teacher.backbone(x_aug)
                logits, _ = self.teacher.sem_seg_head(feats, None)
                logits = F.interpolate(logits, size=(Hs, Ws),
                                       mode="bilinear", align_corners=False)
                probs = logits.float().softmax(dim=1)
                if flip:
                    probs = torch.flip(probs, dims=[-1])
                probs = F.interpolate(probs, size=(H, W),
                                      mode="bilinear", align_corners=False)
                accum = probs if accum is None else accum + probs
                n += 1
        return accum / float(max(n, 1))

    # ------------------------------------------------------------------
    @torch.enable_grad()
    def forward(self, batched_inputs):
        self.iter += 1

        # Trainer.test() puts every submodule into .eval() before running
        # inference_on_dataset; proposal_generator / roi_heads only emit losses
        # when .training is True, so put the student back in train mode here.
        # sem_seg_head is kept in eval() because we consume raw logits.
        self.student.train()
        if not self.det_only and hasattr(self.student, "sem_seg_head"):
            self.student.sem_seg_head.eval()

        # 1. Teacher pseudo-labels.
        pseudo_inst, teacher_sem_results, keep_step = self._teacher_pseudo(batched_inputs)

        # V1: per-task confidence gates. Compute teacher seg confidence
        # independently so each task branch can fire/skip on its own.
        det_gate = keep_step  # default: same as global gate
        seg_gate = not self.det_only  # default: always run unless det_only
        if self.per_task_gate and not self.det_only and not self.seg_only:
            with torch.no_grad():
                # Det gate: skip if teacher det confidence is high (already adapted).
                if len(pseudo_inst[0]) > 0:
                    mean_det_score = float(pseudo_inst[0].scores.mean().item())
                else:
                    mean_det_score = 0.0
                det_gate = mean_det_score < self.per_task_gate_det_thresh
                # Seg gate: skip if teacher seg confidence is high.
                seg_probs_gate = teacher_sem_results.float().softmax(dim=1)
                mean_seg_conf = float(seg_probs_gate.max(dim=1)[0].mean().item())
                seg_gate = mean_seg_conf < self.per_task_gate_seg_thresh

        # V3: cross-task pseudo-label verification.
        if self.ctpv_enabled and len(pseudo_inst[0]) > 0:
            with torch.no_grad():
                t_seg_probs_ctpv = F.interpolate(
                    teacher_sem_results.float(),
                    size=(teacher_sem_results.shape[-2], teacher_sem_results.shape[-1]),
                    mode="bilinear", align_corners=False,
                ).softmax(dim=1)
                pseudo_inst = self._ctpv_filter(pseudo_inst, t_seg_probs_ctpv)

        # 2. Student full forward (backbone -> heads) to get everything we need
        #    in one pass.
        images = self.student.preprocess_image(batched_inputs)
        features = self.student.backbone(images.tensor)

        loss_dict = {}

        if not self.seg_only and det_gate and len(pseudo_inst[0]) > 0:
            # ---- Det consistency: proposal + roi head standard losses
            # on teacher's pseudo boxes.
            gt = self._to_gt_instances(pseudo_inst)
            with EventStorage(self.iter):
                proposals, prop_losses = self.student.proposal_generator(images, features, gt)
                _, det_losses = self.student.roi_heads(images, features, proposals, gt)
            for k, v in prop_losses.items():
                loss_dict[f"det/{k}"] = self.weight_det * v
            for k, v in det_losses.items():
                loss_dict[f"det/{k}"] = self.weight_det * v

        # ---- Sem-seg soft-CE consistency (CoTTA-style).
        if not self.det_only and hasattr(self.student, "sem_seg_head"):
            s_seg_logits, _ = self.student.sem_seg_head(features, None)
        else:
            s_seg_logits = None
        if not self.det_only and seg_gate and s_seg_logits is not None:
            teacher_seg_probs_full = F.interpolate(
                teacher_sem_results.float(), size=s_seg_logits.shape[-2:],
                mode="bilinear", align_corners=False,
            ).softmax(dim=1)
            # CoTTA-style aug-average — trigger on anchor confidence (default)
            # or on TEACHER entropy (E3, addresses CoTTA-1 vulnerability).
            if self.seg_aug_enabled:
                with torch.no_grad():
                    if self.aug_trigger_teacher_entropy:
                        K = teacher_seg_probs_full.shape[1]
                        t_H = -(teacher_seg_probs_full.clamp_min(1e-8) *
                                teacher_seg_probs_full.clamp_min(1e-8).log()).sum(dim=1).mean()
                        norm_H = float(t_H.item()) / math.log(K)
                        trigger = norm_H > self.aug_teacher_entropy_thresh
                    else:
                        anchor_feats = self.anchor.backbone(images.tensor)
                        a_logits, _ = self.anchor.sem_seg_head(anchor_feats, None)
                        a_probs = a_logits.float().softmax(dim=1)
                        conf = a_probs.max(dim=1)[0].mean()
                        trigger = float(conf.item()) < self.seg_aug_conf_thresh
                    if trigger:
                        aug_probs = self._teacher_aug_avg_seg_probs(images.tensor)
                        aug_probs = F.interpolate(
                            aug_probs, size=s_seg_logits.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                        teacher_seg_probs_full = aug_probs
            s_seg_log_probs = F.log_softmax(s_seg_logits.float(), dim=1)
            per_pixel_ce = -(teacher_seg_probs_full.detach() * s_seg_log_probs).sum(dim=1)
            if self.entropy_weighted_ce:
                # E2: down-weight uncertain pixels by (1 - normalized entropy).
                K = teacher_seg_probs_full.shape[1]
                with torch.no_grad():
                    tp = teacher_seg_probs_full.clamp_min(1e-8)
                    t_H_map = -(tp * tp.log()).sum(dim=1)                # (B, H, W)
                    pixel_w = (1.0 - t_H_map / math.log(K)).clamp_min(0.0)
                loss_seg = (pixel_w * per_pixel_ce).sum() / (pixel_w.sum() + 1e-6)
            else:
                loss_seg = per_pixel_ce.mean()
            loss_dict["seg/soft_ce"] = self.weight_seg * loss_seg
        else:
            teacher_seg_probs_full = None

        # ---- Cross-task contrastive (CT-CL). Requires both branches active.
        if (self.ctcl_enabled and not self.det_only and not self.seg_only
                and det_gate and seg_gate
                and len(pseudo_inst[0]) > 0
                and teacher_seg_probs_full is not None):
            loss_ctcl = self._ctcl_loss(features, pseudo_inst, teacher_seg_probs_full)
            loss_dict["ctcl"] = self._cl_boost * self.weight_ctcl * loss_ctcl

        # ---- Cross-task consistency regularizer (CT-CR).
        if (self.weight_ctcr > 0 and not self.det_only and not self.seg_only
                and (det_gate or seg_gate) and len(pseudo_inst[0]) > 0):
            loss_ctcr = self._ctcr_loss(s_seg_logits, pseudo_inst)
            if loss_ctcr is not None:
                loss_dict["ctcr"] = self._cl_boost * self.weight_ctcr * loss_ctcr

        # V4: cross-task prototype anchor loss.
        if (self.proto_anchor and not self.det_only and not self.seg_only
                and len(pseudo_inst[0]) > 0):
            proto_loss = self._proto_anchor_update_and_loss(
                features, pseudo_inst, teacher_seg_probs_full)
            if float(proto_loss.item()) > 0:
                loss_dict["proto"] = self.proto_weight * proto_loss

        # 3. Backward + step.
        if loss_dict:
            total = sum(loss_dict.values())
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.optimizer.step()

        # 4. EMA + stochastic restore.
        self._update_teacher()
        self._stochastic_restore()

        if self.iter % 50 == 0:
            summary = " ".join(f"{k}={float(v.detach()):.3f}" for k, v in loss_dict.items())
            tag = ""
            if self.per_task_gate:
                tag = f" det_gate={det_gate} seg_gate={seg_gate}"
            print(f"[CT-CMT-MTL] iter={self.iter} score_em={self.score_em:.3f} "
                  f"n_pseudo={len(pseudo_inst[0])}{tag} {summary}")

        # 5. Report TEACHER predictions for evaluation (no panoptic combine).
        # 5. Report TEACHER predictions for evaluation (no panoptic combine).
        with torch.no_grad():
            teacher_out = self.teacher.inference(batched_inputs, do_postprocess=False)
            if isinstance(teacher_out, tuple) and len(teacher_out) == 2:
                t_det, t_sem = teacher_out
            else:
                t_det, t_sem = teacher_out, None
        processed = []
        for i, (inp, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
            H = inp.get("height", image_size[0])
            W = inp.get("width", image_size[1])
            det_r = detector_postprocess(t_det[i], H, W)
            out_i = {"instances": det_r}
            if t_sem is not None:
                out_i["sem_seg"] = sem_seg_postprocess(t_sem[i], image_size, H, W)
            processed.append(out_i)
        return processed
