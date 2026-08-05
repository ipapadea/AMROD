# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.optim as optim
from detectron2.config import configurable
from detectron2.layers import move_device_like
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from copy import deepcopy
import math
from .build import META_ARCH_REGISTRY

from collections import OrderedDict
import torch.nn.functional as F
from .losses import  AMRODConLoss
from torchvision.ops import roi_align



__all__ = ["AMROD"]


@META_ARCH_REGISTRY.register()
class AMROD(nn.Module):
    
    @configurable
    def __init__(
        self,
        *,
        model: nn.Module = None,
        model_teacher: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        datasetName: str,
        cfg,
    ):
        super().__init__()
        self.model = model
        self.model_teacher = model_teacher
        self.optimizer = optimizer
        self.steps = 1
        self.model_state_anchor = deepcopy(self.model.state_dict())
        
        self.model_teacher.eval()
        self.model.train()

        self.iter = 0
        self.AMROD_contrastive_loss = AMRODConLoss(temperature=0.07)
        
        self.threshold_init = cfg.SOLVER.THRESHOLD_INIT
        self.mt = cfg.SOLVER.MT
        self.rst_m = cfg.SOLVER.RST_M
        # Backbone/FPN params are general-purpose; protect them more than head params.
        self.backbone_rst_factor = float(getattr(cfg.SOLVER, "CTCMT_BACKBONE_RST_FACTOR", 1.0))
        self.loss_weight = cfg.SOLVER.LOSS_WEIGHT
        self.thresholds_max = cfg.SOLVER.THRESHOLD_MAX
        self.thresholds_mini = cfg.SOLVER.THRESHOLD_MINI
        self.alpha_dt = cfg.SOLVER.ALPHA_DT
        self.gamma_dt = cfg.SOLVER.GAMMA_DT
        self.proposals = cfg.SOLVER.PROPOSALS
        
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.thresholds = [self.threshold_init] * self.num_classes
        self.threshold = 0.9
        dim_in = 1024
        
        self.query_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in)
            )
        self.optimizer.add_param_group({"params": self.query_head.parameters()})
        
        self.value_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_in)
            )
        self.optimizer.add_param_group({"params": self.value_head.parameters()})
        
        self.mean_score = [[] for i in range(self.num_classes)]
        self.thresh = [[] for i in range(self.num_classes)]
        self.last_mean = [0.5 for i in range(self.num_classes)]
        
        self.score_window = []
        self.score_em = cfg.SOLVER.SCORE_EM
        self.score_gamma = cfg.SOLVER.SCORE_GAMMA
        self.score_thresh = cfg.SOLVER.SCORE_THRESH
        self.slope_list = []
        self.stop_count = 0

        # ---- TT-BBR (Test-Time Bounding Box Refinement) -- added 2026-07-28
        # ViTPrompt-inspired second-pass box regression on teacher boxes.
        # See _tt_bbr() for the mechanism.
        self.ttbbr_enabled = getattr(cfg.SOLVER, "TTBBR_ENABLED", False)
        self.ttbbr_iou_thresh = getattr(cfg.SOLVER, "TTBBR_IOU_THRESH", 0.7)
        self.ttbbr_drop = getattr(cfg.SOLVER, "TTBBR_DROP_INCONSISTENT", False)
        # Diagnostic counter: # boxes refined vs # refinements rejected by IoU.
        self._ttbbr_stats = {"refined": 0, "rejected": 0, "frames": 0}

        # ---- XAI-guided pseudo-label filter (Option 1) -- added 2026-07-28
        # Drops teacher boxes whose evidence is spatially diffuse.
        # See _xai_filter() for the three technique variants.
        self.xai_enabled = getattr(cfg.SOLVER, "XAI_FILTER_ENABLED", False)
        self.xai_method = str(getattr(cfg.SOLVER, "XAI_METHOD", "eigencam")).lower()
        self.xai_threshold = float(getattr(cfg.SOLVER, "XAI_THRESHOLD", 0.30))
        self.xai_mode = str(getattr(cfg.SOLVER, "XAI_MODE", "drop")).lower()
        self._xai_stats = {"kept": 0, "dropped": 0, "frames": 0}
        if datasetName == "ACDC":
            self.totalIter = 400
        else:
            self.totalIter = 500
            

        
        
    @classmethod
    def from_config(cls, cfg):
        model = META_ARCH_REGISTRY.get("GeneralizedRCNN")(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        model.to(torch.device(cfg.MODEL.DEVICE))
        
        model_teacher = META_ARCH_REGISTRY.get("GeneralizedRCNN")(cfg)
        DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        model_teacher.to(torch.device(cfg.MODEL.DEVICE))
        for param in model_teacher.parameters():
            param.detach_()
        
        optimizer = build_optimizer(cfg, model)
        
        if cfg.DATASETS.TEST[0] == "c_fog"or cfg.DATASETS.TEST[0] =="fog":
            datasetName = "C"
        elif cfg.DATASETS.TEST[0] == "gaussian_noise":
            datasetName = "C_all"
        elif cfg.DATASETS.TEST[0] == "defocus_blur":
            datasetName = "C_12"
        else:
            datasetName = "ACDC"
        
        return {
            "model": model,
            "model_teacher": model_teacher,
            "optimizer": optimizer,
            "datasetName": datasetName,
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
    
    def KD_loss(self, student_logits, teacher_logits) :
        teacher_prob = F.softmax(teacher_logits, dim=1)
        student_log_prob = F.log_softmax(student_logits, dim=1)
        KD_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction='batchmean')

        return KD_loss
    
    def dynamic_threshold(self, logits_means):
        new_thresholds = [self.gamma_dt * threshold + (1 - self.gamma_dt) * self.alpha_dt * math.sqrt(mean)
                            if mean>0 else threshold for threshold, mean in zip(self.thresholds, logits_means)]
        new_thresholds = [max(min(threshold, self.thresholds_max), self.thresholds_mini) for threshold in new_thresholds]
        return new_thresholds

    # =====================================================================
    # TT-BBR (Test-Time Bounding Box Refinement) -- added 2026-07-28.
    # Motivated by ViTPrompt (Qin et al., CVPR 2026): all prior CTTA-OD
    # methods refine only classification confidence, leaving bounding
    # boxes unchanged. Yet corruption degrades localization independently.
    # ViTPrompt closed this gap for open-vocab detection; we adapt the
    # mechanism to closed-set FR-CNN.
    #
    # Mechanism:
    #   1. Reuse teacher FPN features already computed for initial prediction.
    #   2. Re-pool at teacher's own boxes (feats live in resized-input frame).
    #   3. Run box_head + box_predictor -> per-class delta.
    #   4. Decode to refined boxes.
    #   5. IoU(B_init, B_refined) >= tau -> accept refinement.
    #      Below threshold: keep original box (safe), or drop entirely.
    #   6. Update t_results_raw in place; downstream (dyn thresh, student
    #      loss) sees refined boxes.
    # Cost: one extra RoI-align + box_head + box_predictor call per frame.
    # No weight updates.
    # =====================================================================
    @torch.no_grad()
    def _tt_bbr(self, t_features, t_results_raw):
        if not self.ttbbr_enabled:
            return t_results_raw
        inst = t_results_raw[0]
        if len(inst) == 0:
            return t_results_raw

        # box_pooler expects boxes as list[Boxes] and features as list[Tensor]
        # in the same order as roi_heads.box_in_features.
        roi_heads = self.model_teacher.roi_heads
        feats = [t_features[f] for f in roi_heads.box_in_features]
        boxes_init = inst.pred_boxes  # Boxes in resized-input frame
        pooled = roi_heads.box_pooler(feats, [boxes_init])
        box_feats = roi_heads.box_head(pooled)
        # box_predictor returns (cls_scores, bbox_deltas).
        # bbox_deltas shape: (N, C*4) with class-specific regression, or (N, 4)
        # when reg_class_agnostic=True.
        _, bbox_deltas = roi_heads.box_predictor(box_feats)

        N = boxes_init.tensor.size(0)
        num_classes = self.num_classes
        pred_classes = inst.pred_classes.long()

        if bbox_deltas.size(1) == 4:
            deltas = bbox_deltas
        else:
            # (N, num_classes, 4). Some head layouts have an extra background
            # column (shape (N, (num_classes+1)*4)); handle both.
            deltas = bbox_deltas.view(N, -1, 4)
            n_class_cols = deltas.size(1)
            safe_labels = pred_classes.clamp(max=n_class_cols - 1)
            idx = safe_labels.view(N, 1, 1).expand(N, 1, 4)
            deltas = deltas.gather(1, idx).squeeze(1)  # (N, 4)

        # Decode via the box_predictor's box2box_transform. Clip to image bounds.
        refined_tensor = roi_heads.box_predictor.box2box_transform.apply_deltas(
            deltas, boxes_init.tensor
        )
        # Clip to the RESIZED-input image bounds (image_size = (H, W)).
        H, W = inst.image_size
        refined_tensor[:, 0].clamp_(min=0, max=W)
        refined_tensor[:, 2].clamp_(min=0, max=W)
        refined_tensor[:, 1].clamp_(min=0, max=H)
        refined_tensor[:, 3].clamp_(min=0, max=H)

        # Row-wise IoU (self-consistency filter).
        ious = self._box_pairwise_iou(boxes_init.tensor, refined_tensor)
        consistent = ious >= self.ttbbr_iou_thresh

        # Update stats + apply.
        self._ttbbr_stats["frames"] += 1
        self._ttbbr_stats["refined"] += int(consistent.sum().item())
        self._ttbbr_stats["rejected"] += int((~consistent).sum().item())

        if self.ttbbr_drop:
            keep = consistent
            new_inst = Instances(inst.image_size)
            new_inst.pred_boxes = Boxes(refined_tensor[keep])
            new_inst.pred_classes = inst.pred_classes[keep]
            new_inst.scores = inst.scores[keep]
            # Copy any other fields if present.
            for k in inst.get_fields():
                if k not in ("pred_boxes", "pred_classes", "scores"):
                    new_inst.set(k, inst.get(k)[keep])
            return [new_inst]

        # Non-drop mode: replace boxes where consistent, keep original elsewhere.
        new_tensor = boxes_init.tensor.clone()
        new_tensor[consistent] = refined_tensor[consistent]
        inst.pred_boxes = Boxes(new_tensor)
        return t_results_raw

    @staticmethod
    def _box_pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Row-wise IoU between two aligned (N,4) xyxy tensors."""
        if a.numel() == 0:
            return a.new_zeros((0,))
        inter_x1 = torch.max(a[:, 0], b[:, 0])
        inter_y1 = torch.max(a[:, 1], b[:, 1])
        inter_x2 = torch.min(a[:, 2], b[:, 2])
        inter_y2 = torch.min(a[:, 3], b[:, 3])
        inter = ((inter_x2 - inter_x1).clamp_(min=0) *
                 (inter_y2 - inter_y1).clamp_(min=0))
        area_a = ((a[:, 2] - a[:, 0]).clamp_(min=0) *
                  (a[:, 3] - a[:, 1]).clamp_(min=0))
        area_b = ((b[:, 2] - b[:, 0]).clamp_(min=0) *
                  (b[:, 3] - b[:, 1]).clamp_(min=0))
        union = area_a + area_b - inter
        return inter / union.clamp(min=1e-6)

    # =====================================================================
    # XAI-guided pseudo-label filter (Option 1) -- added 2026-07-28.
    # Drops teacher boxes whose evidence is spatially DIFFUSE. Rationale:
    # under corruption, hallucinated / low-quality detections show smeared
    # feature evidence, while true positives are concentrated on the object.
    #
    # Three technique variants selected via cfg.SOLVER.XAI_METHOD:
    #   "featnorm" : L2 norm of pooled features per spatial location (cheapest;
    #                 no gradient, no SVD). Class-agnostic.
    #   "eigencam" : 1st principal component of the RoI feature tensor
    #                 (Muhammad & Yeasin, 2020). No gradient, one small SVD per
    #                 box. Class-agnostic; sharper than featnorm.
    #   "gradcam"  : gradient of the predicted-class logit x activation, ReLU'd
    #                 (Selvaraju ICCV 2017). One extra backward through
    #                 box_head + box_predictor per frame. Class-specific.
    #
    # All three produce a spatial map (N, H*W); we score each box by
    #   concentration = 1 - H(p) / log(H*W),   p = map / sum(map)
    # in [0,1]. Boxes with concentration < XAI_THRESHOLD are:
    #   - "drop"     : removed from the pseudo-label set.
    #   - "reweight" : score multiplied by concentration (soft filter).
    #
    # Applied AFTER TT-BBR (if enabled) so it filters the refined-box set.
    # Returns (t_results_raw, keep_mask); keep_mask is None if drop mode did
    # not remove any boxes (or if mode == "reweight"), else a 1-D bool tensor
    # over the ORIGINAL t_results_raw[0] indexing so the caller can splice
    # outputs[0]["instances"] similarly.
    # =====================================================================
    @torch.no_grad()
    def _xai_filter(self, t_features, t_results_raw):
        if not self.xai_enabled:
            return t_results_raw, None
        inst = t_results_raw[0]
        N = len(inst)
        if N == 0:
            return t_results_raw, None

        roi_heads = self.model_teacher.roi_heads
        feats = [t_features[f] for f in roi_heads.box_in_features]
        pooled = roi_heads.box_pooler(feats, [inst.pred_boxes])  # (N, C, H, W)
        _, C, H, W = pooled.shape
        HW = H * W
        method = self.xai_method

        if method == "featnorm":
            spatial_map = pooled.norm(dim=1).reshape(N, HW)
        elif method == "eigencam":
            # Batched svd_lowrank on (N, HW, C): 1st left-singular vector is
            # the spatial signature of the top principal component.
            flat = pooled.reshape(N, C, HW).transpose(1, 2)  # (N, HW, C)
            try:
                u, s, v = torch.svd_lowrank(flat, q=1)  # u: (N, HW, 1)
                spatial_map = u.squeeze(-1).abs()
            except Exception:
                spatial_map = pooled.norm(dim=1).reshape(N, HW)
        elif method == "gradcam":
            # Requires autograd. pooled has no grad_fn (teacher inference ran
            # under no_grad), so we clone + require_grad and rerun the head.
            with torch.enable_grad():
                pooled_g = pooled.detach().clone().requires_grad_(True)
                box_feats = roi_heads.box_head(pooled_g)
                cls_scores, _ = roi_heads.box_predictor(box_feats)
                # (N, num_classes + 1) in FR-CNN; select predicted-class logit.
                pred_cls = inst.pred_classes.long().clamp(
                    max=cls_scores.size(1) - 1
                ).view(N, 1)
                target = cls_scores.gather(1, pred_cls).sum()
                grad = torch.autograd.grad(target, pooled_g,
                                            retain_graph=False)[0]
            weights = grad.mean(dim=(2, 3), keepdim=True)      # (N, C, 1, 1)
            cam = (weights * pooled_g).sum(dim=1).clamp(min=0)  # (N, H, W)
            spatial_map = cam.reshape(N, HW)
        else:
            # Unknown method: no-op.
            return t_results_raw, None

        # Normalized entropy -> concentration in [0, 1].
        eps = 1e-8
        spatial_map = spatial_map.detach()
        total = spatial_map.sum(dim=1, keepdim=True) + eps
        p = spatial_map / total
        entropy = -(p * (p + eps).log()).sum(dim=1)
        max_ent = math.log(HW)
        concentration = 1.0 - entropy / max_ent
        # Guard against degenerate (all-zero activation) boxes.
        concentration = torch.nan_to_num(concentration, nan=0.0)
        concentration = concentration.clamp(min=0.0, max=1.0)

        self._xai_stats["frames"] += 1

        # Periodic diagnostic: log concentration percentiles + drop rate so we
        # can calibrate XAI_THRESHOLD without a separate probe run.
        if getattr(self, "iter", 0) % 50 == 0 and N > 0:
            with torch.no_grad():
                c_sorted, _ = concentration.sort()
                q = [0.10, 0.25, 0.50, 0.75, 0.90]
                idx = [max(0, min(N - 1, int(round(qi * (N - 1))))) for qi in q]
                pct = [c_sorted[i].item() for i in idx]
                print(f"[XAI] iter={getattr(self, 'iter', 0)} method={method} "
                      f"N={N} thr={self.xai_threshold:.2f} "
                      f"conc percentiles p10/25/50/75/90="
                      f"{pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f}/{pct[3]:.3f}/{pct[4]:.3f} "
                      f"kept={(concentration >= self.xai_threshold).sum().item()}/{N}")

        if self.xai_mode == "drop":
            keep = concentration >= self.xai_threshold
            self._xai_stats["kept"] += int(keep.sum().item())
            self._xai_stats["dropped"] += int((~keep).sum().item())
            if keep.all():
                return t_results_raw, None
            new_inst = Instances(inst.image_size)
            new_inst.pred_boxes = Boxes(inst.pred_boxes.tensor[keep])
            new_inst.pred_classes = inst.pred_classes[keep]
            new_inst.scores = inst.scores[keep]
            for k in inst.get_fields():
                if k not in ("pred_boxes", "pred_classes", "scores"):
                    new_inst.set(k, inst.get(k)[keep])
            return [new_inst], keep

        if self.xai_mode == "reweight":
            self._xai_stats["kept"] += N
            # Rank-preserving soft filter: multiplicative on scores.
            new_inst = Instances(inst.image_size)
            new_inst.pred_boxes = inst.pred_boxes
            new_inst.pred_classes = inst.pred_classes
            new_inst.scores = inst.scores * concentration.to(inst.scores.dtype)
            for k in inst.get_fields():
                if k not in ("pred_boxes", "pred_classes", "scores"):
                    new_inst.set(k, inst.get(k))
            return [new_inst], None

        # Unknown mode: no-op.
        return t_results_raw, None
    
    def forward(self, x: List[Dict[str, torch.Tensor]]):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, batched_inputs):
        self.iter += 1
        t_features, t_proposals, t_results_raw, outputs = self.model_teacher.inference(batched_inputs, mode= "ctaod")

        # ---- TT-BBR: refine teacher boxes via a second RoI-regression pass.
        # No-op when SOLVER.TTBBR_ENABLED = False. Updates t_results_raw
        # in place (and returns it) so downstream code sees refined boxes.
        # Also update `outputs` (postprocessed, used for CocoMetric eval)
        # so the reported mAP reflects the refined boxes.
        if self.ttbbr_enabled:
            t_results_raw = self._tt_bbr(t_features, t_results_raw)
            # Rescale refined boxes back to the ORIGINAL image frame for eval.
            # `outputs[i].pred_instances` boxes are in original coords; recompute
            # from the (now refined) t_results_raw boxes using the inverse of
            # the resize ratio stored on `outputs`.
            if len(outputs) > 0 and len(t_results_raw[0]) > 0:
                orig_h = outputs[0]["instances"].image_size[0]
                orig_w = outputs[0]["instances"].image_size[1]
                resized_h, resized_w = t_results_raw[0].image_size
                sx = float(orig_w) / float(resized_w)
                sy = float(orig_h) / float(resized_h)
                scale = t_results_raw[0].pred_boxes.tensor.new_tensor(
                    [sx, sy, sx, sy]
                )
                new_boxes = t_results_raw[0].pred_boxes.tensor * scale
                if self.ttbbr_drop:
                    # Rebuild outputs[0]["instances"] with only the surviving
                    # subset since drop mode shrinks the set.
                    new_inst = Instances((orig_h, orig_w))
                    new_inst.pred_boxes = Boxes(new_boxes)
                    new_inst.pred_classes = t_results_raw[0].pred_classes
                    new_inst.scores = t_results_raw[0].scores
                    outputs[0]["instances"] = new_inst
                else:
                    outputs[0]["instances"].pred_boxes = Boxes(new_boxes)

        # ---- XAI-guided pseudo-label filter (Option 1).
        # No-op when SOLVER.XAI_FILTER_ENABLED = False. Applied AFTER TT-BBR
        # so it filters the refined-box set. If drop mode removes boxes, mirror
        # the same mask onto outputs[0]["instances"] for eval consistency.
        if self.xai_enabled:
            t_results_raw, xai_keep = self._xai_filter(t_features, t_results_raw)
            if xai_keep is not None and len(outputs) > 0:
                out_inst = outputs[0]["instances"]
                if len(out_inst) == xai_keep.numel():
                    new_out = Instances(out_inst.image_size)
                    new_out.pred_boxes = Boxes(
                        out_inst.pred_boxes.tensor[xai_keep]
                    )
                    new_out.pred_classes = out_inst.pred_classes[xai_keep]
                    new_out.scores = out_inst.scores[xai_keep]
                    for k in out_inst.get_fields():
                        if k not in ("pred_boxes", "pred_classes", "scores"):
                            new_out.set(k, out_inst.get(k)[xai_keep])
                    outputs[0]["instances"] = new_out

        valid_map = t_results_raw[0].scores > 0.1
        valid_score = t_results_raw[0].scores[valid_map]

        if valid_map.sum() == 0:
            self.stop_count += 1
            return outputs
        else:
            mean_all = valid_score.mean().cpu()
            if self.iter % 50 == 0:
                print("iter: ", self.iter, "score_em", self.score_em)
            if (mean_all / self.score_em) > self.score_thresh or (self.score_em / mean_all) > self.score_thresh:
                self.stop_count += 1
                self.score_em = self.score_gamma * self.score_em + (1-self.score_gamma) * mean_all
                return outputs
            self.score_em = self.score_gamma * self.score_em + (1-self.score_gamma) * mean_all
                
        score_mean = [torch.zeros(1, dtype=t_results_raw[0].scores.dtype, device=t_results_raw[0].scores.device) for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            index = t_results_raw[0].pred_classes[valid_map] == i
            if index.sum() > 0:
                score_mean[i] = valid_score[index].mean()
        
        self.thresholds = self.dynamic_threshold(score_mean)
        t_results = process_pseudo_label(t_results_raw, self.thresholds, True)

            
        loss_dict = {}       
        images = self.model.preprocess_image(batched_inputs, strong_aug = True)
        features = self.model.backbone(images.tensor)
        proposals, proposal_losses = self.model.proposal_generator(images, features, t_results)
        _, detector_losses = self.model.roi_heads(images, features, proposals, t_results)
        
        loss_dict.update(detector_losses)
        loss_dict.update(proposal_losses)
        
        t_proposals[0].proposal_boxes.tensor = t_proposals[0].proposal_boxes.tensor[:self.proposals]
                
        features = [features[f] for f in self.model.roi_heads.box_in_features]
        s_box_features = self.model.roi_heads.box_pooler(features, [x.proposal_boxes for x in t_proposals])
        s_box_features = self.model.roi_heads.box_head(s_box_features)
        s_roih_logits = self.model.roi_heads.box_predictor(s_box_features)
        
        t_features = [t_features[f] for f in self.model_teacher.roi_heads.box_in_features]
        t_box_features = self.model_teacher.roi_heads.box_pooler(t_features, [x.proposal_boxes for x in t_proposals])
        t_box_features = self.model_teacher.roi_heads.box_head(t_box_features)
        t_roih_logits = self.model_teacher.roi_heads.box_predictor(t_box_features)
        

        s_query = self.query_head(s_box_features)
        t_query = self.query_head(t_box_features)
        loss_dict["AMROD"] = self.loss_weight * self.AMROD_contrastive_loss(s_query, t_query)
        
        loss_dict["st_const"] = self.KD_loss(s_roih_logits[0], t_roih_logits[0])  
        
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict        
        self.optimizer.zero_grad()
        losses.backward()
        
        if self.rst_m>0:
            fisher_dict = {}
            for nm, m  in self.model.named_modules():  ## previously used model, but now using self.model
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad and p.grad is not None:
                        fisher_dict[f"{nm}.{npp}"] = p.grad.data.clone().pow(2)
        
        self.optimizer.step()
        
        if self.iter % 50 == 0:
            print("iter: ", self.iter, ''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()]))
            print("iter: ", self.iter, "thresholds: ", self.thresholds)
            print()
            
        self.model_teacher = update_model(self.model, self.model_teacher, self.mt)
        
        for nm, m  in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    if f"{nm}.{npp}" not in fisher_dict:
                        continue
                    data = fisher_dict[f"{nm}.{npp}"]
                    mask = find_weight_quantile(data, self.rst_m)
                    # V2: scale restore probability for shared backbone/FPN params.
                    if self.backbone_rst_factor < 1.0:
                        key = f"{nm}.{npp}"
                        _shared = ("backbone.", "fpn.", "proposal_generator.anchor_generator.")
                        if any(key.startswith(pfx) for pfx in _shared):
                            mask = mask * self.backbone_rst_factor
                    with torch.no_grad(): 
                        p.data = self.model_state_anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)               
            
        return outputs
    
def find_weight_quantile(matrix, perc):    
    weights = matrix / matrix.max()
    weights_with_noise = weights * torch.rand_like(weights)
    
    arr_sorted = torch.sort(weights_with_noise.reshape(-1)).values
    frac_idx = perc*(len(arr_sorted)-1)
    frac_part = frac_idx - int(frac_idx)
    low_idx = int(frac_idx)
    high_idx = low_idx + 1
    threshold = arr_sorted[low_idx] + (arr_sorted[high_idx]-arr_sorted[low_idx]) * frac_part # linear interpolation
    mask = weights_with_noise < threshold
    return mask.float().cuda()


def threshold_bbox(proposal_bbox_inst, thres, dynamic = False):
    
    if dynamic:
        thres = torch.tensor(thres).to(proposal_bbox_inst.scores.device)
        valid_map = torch.gt(proposal_bbox_inst.scores, thres[proposal_bbox_inst.pred_classes])
        
    else:
        valid_map = proposal_bbox_inst.scores > thres

    # create instances containing boxes and gt_classes
    image_shape = proposal_bbox_inst.image_size
    new_proposal_inst = Instances(image_shape)

    # create box
    new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
    new_boxes = Boxes(new_bbox_loc)

    # add boxes to instances
    new_proposal_inst.gt_boxes = new_boxes
    new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
    new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst


def process_pseudo_label(proposals_rpn_unsup_k, cur_threshold, dynamic = False):
    list_instances = []
    for proposal_bbox_inst in proposals_rpn_unsup_k:
        # thresholding
        proposal_bbox_inst = threshold_bbox(
            proposal_bbox_inst, cur_threshold, dynamic
        )
        list_instances.append(proposal_bbox_inst)
        
    return list_instances


@torch.no_grad()
def update_model(model_student, model_teacher, keep_rate):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )   
        else:
            raise Exception("{} is not found in student model".format(key))

    model_teacher.load_state_dict(new_teacher_dict)
    return model_teacher

