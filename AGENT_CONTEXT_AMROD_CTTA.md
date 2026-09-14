# AMROD / CT-CMT-MTL — Agent Context and Research Status

**Status snapshot:** 2026-09-08  
**Repository:** `ipapadea/AMROD`  
**Primary branch:** `extension-implementation`  
**Main working directory:** `~/AMROD`

> This document is the recommended entry point for any coding or research agent working on the repository.
> It explains what the repository is, the research goal, what has already been implemented and tested, what is still unresolved, and which files should be treated as implementation ground truth.
>
> Older Markdown files in the repository contain useful historical material, but some are stale or contradict the current code. When there is a conflict, **the current implementation and this document take precedence**.

---

# 1. Research objective

The project studies **continual test-time adaptation (CTTA)** for a **multi-task perception model** that performs:

- **2D object detection**
- **semantic segmentation**

from a **shared backbone/FPN**, under a continually changing target stream and **without access to target-domain labels during adaptation**.

The target paper extends the earlier single-task CT-CMT work:

> Moraiti et al., *Continual Test-Time Domain Adaptation for Object Detection via Contrastive Mean Teacher and Stochastic Restoration* (2026).

The new paper asks:

> Can cross-task signals between detection and semantic segmentation improve continual adaptation, reduce pseudo-label error propagation, and reduce catastrophic forgetting compared with single-task CTTA methods?

The paper must show that a multi-task CTTA method:

1. adapts both tasks online,
2. remains stable over long streams,
3. exploits cross-task information in a controlled and measurable way,
4. is competitive with strong single-task CTTA baselines,
5. does not obtain gains from accidental implementation confounds,
6. preserves source knowledge under continual drift.

---

# 2. Source models and tasks

Primary source dataset:

- **Cityscapes**

Main multi-task source model:

- **Panoptic FPN R-50**
- shared ResNet-50 + FPN
- detection branch
- semantic segmentation branch
- 8 Cityscapes thing classes for detection
- 19 semantic trainIds

Single-task specialist source models also exist:

- Mask R-CNN R-50 FPN — detection
- Semantic FPN R-50 — semantic segmentation

Comparisons across different source architectures must be labeled clearly as cross-architecture or specialist-vs-unified.

---

# 3. Main CTTA protocols

## ACDC continual

Current controlled stream:

```text
fog → night → rain → snow
```

Properties:

- one continual model instance
- no reset between domains
- online adaptation
- batch size effectively 1
- detection AP/AP50 and semantic mIoU

A final paper protocol should additionally include **clean Cityscapes loopback / source-retention evaluation in no-update mode**. If adaptation is allowed during loopback, forgetting is underestimated.

## Cityscapes-C mixed long-term

Current main long-term stream:

```text
(fog → motion_blur → snow → brightness → defocus_blur) × 10
```

This gives:

- 50 sequential evaluations
- no reset
- one persistent continual model
- detection AP50
- semantic mIoU

This is currently the most important stress test for long-term interference and forgetting.

## Repeated fog ×10 diagnostic

A separate diagnostic stream is:

```text
fog × 10
```

This is **not the main benchmark**. Its purpose is to distinguish repeated same-domain drift from interference caused by domain switching.

---

# 4. Implementation ground truth

The most important implementation file is:

```text
detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py
```

Also inspect:

```text
detectron2/detectron2/config/defaults.py
detectron2/configs/Cityscapes/*.yaml
detectron2/detectron2/data/datasets/builtin.py
```

Agents should inspect these files directly before making methodological claims.

---

# 5. Mean-teacher adaptation skeleton

The MTL adapter uses:

- **student**
- **EMA teacher**
- **frozen source anchor**

At each test image:

1. teacher predicts detection and segmentation outputs,
2. detection pseudo-labels are filtered using dynamic thresholds,
3. a score-EMA gate may decide whether adaptation is needed,
4. student detection/segmentation losses are computed,
5. cross-task losses may be added,
6. optimizer step,
7. teacher EMA update,
8. stochastic restoration of student parameters,
9. prediction is produced from the updated state.

The implementation is effectively **adapt-then-predict**.

The intended setting is:

- source-free,
- online,
- continual,
- batch-1 compatible,
- detector-agnostic where possible.

---

# 6. Difference from the original CT-CMT paper

The current multi-task implementation is **not an exact reproduction of every mechanic in the prior CT-CMT paper**.

Important differences include:

- dynamic pseudo-label thresholds,
- score-EMA gating,
- Panoptic FPN / Detectron2 architecture,
- current CT-CL construction,
- semantic segmentation consistency,
- CT-CR,
- task-aware restoration,
- CTPV.

Do not describe all current mechanics as if they were directly inherited from the prior paper.

---

# 7. Current CT-CL implementation

CT-CL uses student FPN features.

## Detection view

```text
RoIAlign on student FPN feature
→ spatial mean
→ L2 normalization
```

## Segmentation view

The teacher semantic posterior for the mapped semantic class is used as a spatial weighting mask over the **student FPN feature crop** inside the detection box.

The weighted feature is pooled and L2-normalized.

The combined views are used in a supervised contrastive objective using the detection pseudo-class as label.

### Important

The current code uses **raw FPN feature vectors**.

There are currently **no learned 128-dimensional projection heads** in CT-CL.

Older docs that describe explicit learned `phi_det` / `phi_seg` projection heads are stale.

---

# 8. Current stochastic restoration / V2

The config flag is historically named:

```text
CTCMT_CROSS_TASK_FISHER
```

but the current V2 mechanism is **not a Fisher-information method**.

Current V2 is better described as:

> **task-aware / backbone-protected stochastic restoration**

Let:

- base restoration probability = `p`
- backbone factor = `η`

Then:

```text
task-specific heads: restore with probability p
shared backbone/FPN: restore with probability p × η
```

Typical setting:

```text
p = 0.01
η = 0.1
```

Therefore:

```text
shared restore probability = 0.001
head restore probability   = 0.01
```

The shared trunk is restored **10× less often**, not 10× more often.

Interpretation:

> V2 protects shared backbone/FPN adaptation from being repeatedly erased.

Do not describe V2 as Fisher restoration unless a real Fisher-based mechanism is implemented.

---

# 9. CT-CR research question

CT-CR propagates detection-side class information into the semantic branch.

The research question is:

> How should detection boxes supervise semantic pixels without injecting excessive full-box label noise?

Four formulations are now being compared.

## A — legacy full-box CT-CR

For each pseudo detection:

- map detection class to semantic class,
- assign that class to the entire rectangle,
- build a global semantic target map,
- overlapping boxes can overwrite earlier labels,
- compute semantic cross-entropy over the target map.

This is the legacy reference.

## A2 — per-box full CT-CR

A2 is a **control ablation**.

It keeps the same full-box semantic assumption as A, but changes only aggregation:

\[
L_{A2}
=
rac{1}{N}
\sum_j
rac{1}{|B_j|}
\sum_{i\in B_j}
CE(p_i,c_j)
\]

Properties:

- full box supervised,
- no semantic mask,
- no semantic probability weighting,
- each box normalized independently,
- equal mean across boxes,
- overlapping boxes evaluated independently,
- no global overwrite.

Purpose:

> Separate the effect of semantic spatial weighting from the effect of changing the loss aggregation.

Current mode name:

```text
CTCMT_CTCR_MODE = "per_box_full"
```

At this snapshot, A2 experiments are running / queued.

## B — hard semantic-support CT-CR

Inside each box:

```text
q(i) = detached teacher semantic probability
       for the mapped detection class
```

Only pixels satisfying:

\[
q(i) \geq 0.3
\]

are supervised.

Loss is normalized per box and averaged across boxes.

## C — soft semantic-support CT-CR

Every box pixel remains, but is weighted:

\[
w(i)=lpha+(1-lpha)q(i)
\]

Current setting:

\[
lpha=0.2
\]

Loss is normalized per box and averaged across boxes.

---

# 10. Why CT-CR was redesigned

A GT-aware diagnostic showed substantial full-box label noise.

Approximate semantic purity inside pseudo-detection boxes:

```text
mean   ≈ 0.716
median ≈ 0.771
```

Lower-purity classes included approximately:

```text
rider       ≈ 0.45
bicycle     ≈ 0.53
motorcycle  ≈ 0.59
person      ≈ 0.60
```

This motivated hard and soft segmentation-supported CT-CR.

However, reducing label noise does **not automatically produce better continual adaptation**.

---

# 11. Current A/B/C results

## Repeated fog ×10 — seed 0

| Variant | mean AP50 | mean mIoU |
|---|---:|---:|
| A — full box | 41.0037 | 43.7876 |
| B — hard support | 41.8192 | 43.8318 |
| **C — soft support** | **43.1046** | **45.2351** |

Interpretation:

> Soft semantic weighting is clearly beneficial when the target domain remains stable.

This is a diagnostic result, not sufficient by itself to select C as the final method.

## ACDC — 3 seeds

| Variant | AP50 mean ± std | mIoU mean ± std |
|---|---:|---:|
| **A — full box** | **37.343 ± 0.047** | **34.206 ± 0.108** |
| B — hard support | 36.863 ± 0.120 | 33.964 ± 0.095 |
| C — soft support | 36.678 ± 0.136 | 33.754 ± 0.163 |

Important progressive pattern for C vs A in AP50:

```text
fog   ≈ +0.23
night ≈ +0.02
rain  ≈ -0.91
snow  ≈ -2.00
```

Interpretation:

> Soft semantic weighting is competitive early in ACDC, but its disadvantage relative to full-box supervision grows later in the continual stream.

Possible explanation: semantic reliability becomes stale or cross-task interference accumulates after domain switches. This remains a hypothesis.

## Cityscapes-C mixed long-term ×10 — 3 seeds

### A

```text
seed 0   AP50 18.9588   mIoU 29.0578
seed 42  AP50 18.1980   mIoU 29.1733
seed 123 AP50 19.0124   mIoU 29.0696
```

### B

```text
seed 0   AP50 17.7088   mIoU 29.1802
seed 42  AP50 18.5235   mIoU 29.1315
seed 123 AP50 19.1385   mIoU 29.4846
```

### C

```text
seed 0   AP50 18.0060   mIoU 29.1580
seed 42  AP50 19.1260   mIoU 29.2507
seed 123 AP50 19.4198   mIoU 29.6621
```

Three-seed summary:

| Variant | AP50 mean ± std | mIoU mean ± std |
|---|---:|---:|
| A | 18.723 ± 0.456 | 29.100 ± 0.064 |
| B | 18.457 ± 0.717 | 29.265 ± 0.191 |
| **C** | **18.851 ± 0.746** | **29.357 ± 0.268** |

Interpretation:

- C has the best mean AP50 and mIoU.
- C consistently beats B.
- C vs A AP50 is **not stable across seeds**.
- The mean AP50 advantage of C over A is much smaller than seed variance.
- C looks more consistently favorable for semantic mIoU than for detection.

Therefore:

> Current evidence does not justify claiming that C is clearly superior to A overall.

---

# 12. Hardware reproducibility issue

The mixed-stream seeds were initially split across:

- Cronus: RTX 3090
- gpu1: NVIDIA L40S

Because online CTTA trajectories are numerically sensitive, A seed42 is being repeated on L40S.

Desired controlled layout:

```text
seed 0:
A / A2 / B / C on RTX 3090

seed 42:
A / A2 / B / C on L40S

seed 123:
A / A2 / B / C on L40S
```

Do not over-interpret tiny differences before this correction is incorporated.

---

# 13. Current CTPV implementation

Current CTPV is a **hard pseudo-detection filter**.

For each detection box:

1. take teacher semantic probabilities,
2. compute semantic argmax per pixel,
3. calculate the fraction of pixels whose argmax equals the semantic class corresponding to the detection class,
4. keep the detection if the fraction is above threshold.

Current threshold:

```text
0.3
```

Mathematically:

\[
agreement(b,c)
=
rac{
|\{i\in b:rg\max_k p_{i,k}=\sigma(c)\}|
}{
|b|
}
\]

Keep if:

\[
agreement \geq 0.3
\]

Older docs that describe CTPV as the **mean semantic posterior** in the box are stale.

---

# 14. CTPV diagnostic result

A GT-aware diagnostic on Cityscapes-C fog showed:

Before CTPV:

```text
TP = 3523
FP = 212
precision ≈ 94.32%
```

After current CTPV at 0.3:

```text
TP = 3396
FP = 200
precision ≈ 94.44%
```

Rejected:

```text
TP = 127
FP = 12
```

Most rejected true positives were not rejected because the box geometry was poor. The dominant failure was:

> segmentation disagreed with a detection that was actually correct.

Thus current CTPV often behaves as a **semantic veto** on correct detections.

Historical ACDC experiments still showed a small end-to-end gain from CTPV, so it should not be called universally harmful.

Current conclusion:

> Binary CTPV is unresolved and should probably be redesigned into a softer reliability-aware cross-task coupling rather than kept as a hard whole-detection veto.

---

# 15. Relationship to single-task baselines

## ACDC detection

Existing single-task results:

```text
Source-only Mask R-CNN      ≈ 32.87 AP50
W3TTA                       ≈ 34.20
AMROD                       ≈ 35.34
AMROD + V2                  ≈ 36.06
CTCMT_Det                   ≈ 36.26
```

Detection-side adaptation is competitive on ACDC.

## ACDC semantic segmentation

Existing specialist results:

```text
Source Semantic FPN   ≈ 36.78 mIoU
CoTTA                 ≈ 36.39
CTCMT_Seg             ≈ 37.67
```

Single-task segmentation is also competitive.

## Cityscapes-C long-term detection

This is currently the largest weakness.

Historical means:

```text
AMROD                 ≈ 27.30 AP50
CTCMT_Det             ≈ 21.94
old CTCMT_MTL         ≈ 16.86
historical final-ish
MTL + V2 + CTPV       ≈ 17.50
```

Current A/B/C mixed MTL variants are around:

```text
18–19 AP50
```

Therefore:

> The multi-task method has a large detection gap to AMROD on Cityscapes-C mixed long-term.

This gap is too large to dismiss as noise.

Future analysis should determine whether it comes from:

- weaker MTL source representation,
- shared-backbone negative transfer,
- CT-CL,
- CT-CR,
- restoration dynamics,
- pseudo-label instability,
- domain-switch interference,
- or several of these together.

---

# 16. Methodological constraint for future ideas

The final method should remain as detector-agnostic as reasonably possible.

In particular:

> Do not rely on instance masks as a necessary component.

The method should remain compatible in principle with detectors without instance masks, including YOLO-family detectors.

Semantic segmentation may provide spatial support, but instance segmentation should not be required.

---

# 17. Experiment status

## Completed

### CT-CR A/B/C

- ACDC, seeds 0 / 42 / 123
- Cityscapes-C mixed LT ×10, seeds 0 / 42 / 123
- repeated fog ×10, seed 0

### Diagnostics

- CTPV GT-aware rejection analysis
- full-box semantic purity analysis
- hard CT-CR threshold sweep
- repeated-fog stability test
- mixed long-term per-domain / per-round analysis

## Running / queued at this snapshot

### A2 control

- A2 mixed LT seed0 — Cronus / RTX 3090
- A2 mixed LT seed42 — gpu1 / L40S
- A2 mixed LT seed123 — gpu1 / L40S
- A2 ACDC seed0
- A2 ACDC seed42
- A2 ACDC seed123
- A2 fog×10 seed0

### Hardware correction

- legacy A mixed LT seed42 rerun on L40S

Once these finish, **gather results before making another CT-CR design change**.

---

# 18. Immediate decision after A2

A2 answers:

> Are the differences between A and B/C caused by semantic masking/weighting, or simply by changing loss aggregation?

Interpretation patterns:

## Case 1: `A ≈ A2`

Then aggregation is minor. Differences from A2 to B/C can be attributed more confidently to semantic support.

## Case 2: `A` significantly differs from `A2`

Then prior A-vs-B/C conclusions were confounded by aggregation.

## Case 3: `A2 > A` and `C > A2`

Then both per-box aggregation and soft semantic weighting provide independent gains.

---

# 19. What should happen after CT-CR is locked

Recommended order:

1. **Finalize CT-CR** using A / A2 / B / C.
2. **Redesign cross-task reliability / CTPV** toward soft modulation rather than hard veto.
3. **Add clean Cityscapes no-update loopback** for source forgetting.
4. **Explain the Cityscapes-C detection gap** to AMROD.
5. **Build a clean additive ablation**:
   ```text
   MTL mean-teacher baseline
   → + CT-CL
   → + final CT-CR
   → + V2 task-aware restoration
   → + final reliability mechanism
   ```
6. **Run final baseline suite** for detection and segmentation.
7. Add runtime, memory, per-class, sensitivity, alternative order, and statistical analyses after the core method is stable.

---

# 20. Documentation warning

The following files contain useful historical context but should **not automatically be treated as current truth**:

```text
METHOD.md
EXTENSION.md
RESEARCH_LOG.md
FUTURE_WORK.md
USAGE.md
RUNBOOK.md
.history/*.md
```

Known stale themes include:

- projection heads described although current CT-CL uses raw FPN features,
- V2 restore direction described incorrectly,
- V2 called Fisher although current mechanism is not Fisher-based,
- CTPV described as mean posterior rather than hard argmax-agreement fraction,
- older “best config” claims,
- future-work items already completed,
- old result tables predating current 3-seed A/B/C experiments.

Use these files for chronology and ideas, not as implementation ground truth.

---

# 21. Recommended instructions for a brainstorming agent

A research agent should:

1. read this document first,
2. inspect `ctcmt_mtl.py`,
3. inspect active configs,
4. separate **verified results** from **hypotheses**,
5. identify experimental confounds before proposing new methods,
6. prioritize explanations of major weaknesses over tiny metric tuning,
7. preserve the source-free online continual setting,
8. preserve detector-agnosticity where possible,
9. not assume instance masks are available,
10. propose experiments with explicit causal questions.

Good brainstorming questions:

- Why does soft CT-CR help repeated fog but lose on later ACDC domains?
- Is semantic confidence stale after domain transitions?
- Does CT-CR induce task conflict in the shared backbone?
- Why is Cityscapes-C mixed detection so far behind AMROD?
- Which gradients dominate the shared trunk over long streams?
- Does CT-CL help or hurt under severe corruption?
- Can disagreement be used as a soft reliability signal rather than a veto?
- Can shared-trunk restoration probability adapt to measured cross-task conflict?
- Is the source MTL model itself limiting Cityscapes-C detection?
- How much of the long-term gap remains if cross-task losses are disabled?

---

# 22. Current high-level scientific picture

The project has established several useful facts:

1. Multi-task CTTA improves over the MTL source model on ACDC.
2. Task-aware restoration is useful.
3. Full-box CT-CR contains substantial label noise.
4. Semantic spatial support is informative.
5. Hard semantic filtering is too aggressive.
6. Soft semantic weighting helps strongly under a stable repeated domain.
7. Soft weighting is not clearly superior under continually changing domains.
8. Cross-task semantic veto can reject correct detections.
9. Long-term mixed-domain interference is more serious than repeated same-domain drift.
10. Cityscapes-C long-term detection is currently the largest performance weakness relative to strong single-task CTTA.

The paper should move away from the simplistic claim:

> “more cross-task agreement is always better”

and toward the more defensible research question:

> **When is cross-task information reliable enough to help continual adaptation, and how should that reliability modulate adaptation without amplifying errors from one drifting task into the other?**

That is currently the most promising direction for the final contribution.
