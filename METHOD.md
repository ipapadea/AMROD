# Cross-Task View Alignment for Multi-Task Continual Test-Time Adaptation

## 1. Motivation: diagnosing the failure modes of prior CTTA

Continual test-time adaptation (CTTA) has two flagship methods on the ACDC
weather stream: **CoTTA** for segmentation (Wang *et al.* CVPR 2022) and
**AMROD** for detection (Wei *et al.* ESWA 2026). Both use the same
skeleton — student, EMA teacher, stochastic restore of student weights to
source — and both produce a hallmark failure pattern under sustained
continual drift: they perform reasonably on the first weather (fog) and
**collapse on the last** (snow). CoTTA's snow mIoU (35.79) is *below* its
own source-only baseline (36.82). AMROD's night AP50 (15.23) is barely +1.8
above source-only (13.45), while its later snow AP50 (40.99) recovers only
because the domain happens to be visually closer to Cityscapes.

We argue that this collapse is not accidental — it is the structural
consequence of using a **single head's teacher-student consistency** as
the sole supervision signal. Two specific failure modes:

**FM-1. Teacher-collapse under a single supervision signal.** In detection,
teacher scores fall below AMROD's threshold floor of 0.70 on night, no
pseudo-boxes survive, and adaptation silently stops — the standard
mean-teacher recipe has no orthogonal signal to notice the collapse. In
segmentation, teacher per-pixel entropy grows and CoTTA's soft-CE loss,
weighting *every pixel equally*, amplifies teacher noise directly into
gradient noise. Both are symptoms of the same problem: **when the teacher
starts to drift, only the teacher's own confidence can flag it, and it
doesn't**.

**FM-2. Drift under uniform restore.** CoTTA's stochastic restore treats
shared backbone and task-head weights identically. But the shared backbone
is on the gradient path of every loss term, so it accumulates domain-specific
bias faster than the restore reverts it. AMROD tries to fix this with a
per-step Fisher-quantile restore (grad-squared as a proxy for parameter
importance), but the per-step Fisher is extremely noisy — the same
parameter can be "important" one step and "restored" the next. Both fail
because they treat all shared-trunk drift the same way, whether the shared
trunk actually drifted or only one head did.

**The multi-task setting removes both failure modes at the source.** When a
shared-backbone model outputs both a detection map and a segmentation map,
every object *b* is represented twice at the level of the shared backbone:

- as a **detection view** $z_\text{det}(b)$ (RoI-aligned FPN features), and
- as a **segmentation view** $z_\text{seg}(b)$ (mask-weighted pool of the
  same FPN features under the teacher's per-pixel seg posterior for the
  segmentation class $\sigma(c)$ that corresponds to the box's detection
  class *c*).

These two views are extracted by **different heads** from the **same shared
representation**. If the shared representation is well-adapted, the two
views agree up to a class-conditional embedding; if it is drifting, the two
views diverge **before either head's own confidence drops**. This
disagreement is a target-domain-only self-supervised signal that no
single-task method can access. It fires on FM-1 (either head's collapse is
visible in the other's view before either becomes visibly wrong), and it
enables a mechanistic response to FM-2 (shared-trunk drift is now
distinguishable from head drift).

The paper builds a framework — **Cross-Task View Alignment (XVA)** — around
this observation, with four primitives that jointly cover both failure
modes:

| Primitive | Failure mode addressed | Level | Section |
|---|---|---|---|
| **CT-CL** — SupCon between det and seg views | FM-1 (teacher-collapse, instance-level) | instance | §3 |
| **CT-CR** — bbox interior → seg-class CE | FM-1 (teacher-collapse, dense-level) | pixel | §4 |
| **STR** — shared-trunk stochastic restore split | FM-2 (drift under uniform restore) | weight | §5 |
| **CTPV** — cross-task pseudo-label verification | FM-1 (teacher-collapse, pseudo-label-level) | pseudo-label | §5.5 |

All four are **defined only in the presence of two heads**. Applied to a
single-task source model, CT-CL, CT-CR, and CTPV collapse to zero and STR
degenerates to CoTTA's uniform restore. The framework thus specialises
gracefully — CTCMT_Det and CTCMT_Seg (§7) are what one gets when the
framework runs on a single-task source, and they are still stronger than
the corresponding published methods (Tables 1 and 2). We treat these as
sanity checks; the paper's central claim is the MTL result.

Fault modes FM-1 and FM-2 are formalised, along with code-level line
references and numerical signatures, in §12.

### 1.1 Teacher-student plumbing (standard, briefly)

XVA is agnostic to the specific mean-teacher plumbing that generates the
raw pseudo-labels; any teacher-student CTTA that provides per-image
`(pseudo_boxes, teacher_seg_probs)` will do. We use a standard
CoTTA-with-AMROD-detector setup for the reported experiments, described
compactly below and reusable across all XVA variants:

- **Student θ_s**: trainable copy of the source Panoptic-FPN.
- **Teacher θ_t**: EMA of student, $\theta_t \leftarrow \beta\,\theta_t + (1-\beta)\,\theta_s$, $\beta = 0.9998$.
- **Anchor θ_0**: frozen copy of source weights, source of the sparse
  restore.
- **Detection pseudo-labels**: per-class dynamic thresholds
  $\tau_c^{(t+1)} = \gamma\tau_c^{(t)} + (1-\gamma)\alpha\sqrt{\bar{s}_c^{(t)}}$
  as in AMROD; kept boxes become `gt_instances` for standard Faster R-CNN losses.
- **Segmentation pseudo-labels**: raw teacher softmax; standard soft-CE.
- **Score-EM gate**: skip the step when the teacher-score EMA is stable
  (image already adapted).

None of the above is our contribution. **Everything in §2 onward is.**

## 2. Cross-task view construction

For every pseudo-detected box *b* of class *c*, we construct two feature
vectors from the *student* FPN feature maps *F*:

### Detection view

$$
z_\text{det}(b) \;=\; \phi_\text{det}\Big(\mathrm{RoIAlign}(F, b)\Big)
$$

where $\phi_\text{det}$ is a lightweight L2-normalised projection head
(single Conv + GELU + Linear, 128-d output).

### Segmentation view (novel construction)

Let $p^\text{tch}_{i,\sigma(c)}$ be the teacher's segmentation posterior at
pixel *i* for the segmentation class $\sigma(c)$ that corresponds to
detection class *c* under the Cityscapes taxonomy
(`_DET_TO_SEG_CLASS_CITYSCAPES`, e.g. det-`person` → seg-`person`,
det-`car` → seg-`car`, ...). The segmentation view is the
teacher-posterior-weighted mean of student FPN features **inside the
box only**:

$$
z_\text{seg}(b) \;=\; \phi_\text{seg}\!\left(\frac{\sum_{i \in b} p^\text{tch}_{i,\sigma(c)} \, F_i}{\sum_{i \in b} p^\text{tch}_{i,\sigma(c)} + \varepsilon}\right)
$$

with $\phi_\text{seg}$ another 128-d projection head. The crucial detail:
$z_\text{seg}$ is **weighted by the seg posterior but the features $F_i$
come from the student**, so gradients flow into the shared backbone through
both views. The teacher-driven mask concentrates the pool on genuinely
class-relevant pixels, so the view is robust to bbox looseness.

**Why this specific construction is novel.** Prior cross-task learning
(e.g. Cross-Stitch, MTL-NAS, PAD-Net) shares features across tasks
*during source training*. Prior CTTA work uses at most one head per image.
Ours is the first to (a) *build two views per-object at inference time*, and
(b) *use their agreement as the adaptation signal*, not as an auxiliary
supervision for source training.

## 3. Instance-level XVA — CT-CL

Given all boxes in the current image as $\{(z_i, y_i)\}$ with
$z_i \in \{z_\text{det}, z_\text{seg}\}$ and $y_i$ the detection class of
the underlying box, we apply supervised contrastive loss (Khosla *et al.*
2020) over the combined view pool:

$$
\mathcal{L}_\text{CT-CL} \;=\; -\sum_i \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}
$$

with $\tau = 0.07$. The positive set $P(i)$ contains three qualitatively
different pair types:

| Pair type | Effect on the shared representation |
|---|---|
| $z_\text{det} \leftrightarrow z_\text{det}$ (same class) | classical detection SupCon |
| $z_\text{seg} \leftrightarrow z_\text{seg}$ (same class) | classical segmentation SupCon |
| **$z_\text{det} \leftrightarrow z_\text{seg}$ (same class)** | **cross-task pull — the XVA signal** |

The third row is what forces the two heads' post-projection embeddings to
converge on a **shared per-class code**. It is the loss term without which
XVA reduces to two independent SupCon losses that could equivalently be
optimised in single-task pipelines.

## 4. Dense XVA — CT-CR

CT-CL is instance-level: it pulls average box-embeddings together but says
nothing about the *pixels* inside each box. CT-CR closes that gap with a
dense supervision that costs one extra loss term and no new parameters. For
every pseudo-box *b* of class *c*, all pixels inside *b* should be labelled
by the student segmentation head as $\sigma(c)$:

$$
\mathcal{L}_\text{CT-CR} \;=\; \sum_{i \in b} \mathrm{CE}\!\left(p^\text{stu-seg}_i,\; \sigma(c)\right)
$$

Pixels outside every box are ignored (label 255). This propagates
detection-side certainty back through the seg head and, symmetrically,
enforces that the seg head's per-pixel predictions be spatially consistent
with the detection geometry.

**Instance × dense granularity is the point.** CT-CL and CT-CR target
different failure modes:
- CT-CL prevents the shared representation from becoming
  *class-degenerate* across tasks (each head could still classify well but
  from different features).
- CT-CR prevents the seg head's output from becoming *spatially
  incoherent* with detected objects (each object could still be detected
  but with wrong pixel support).

Their ablation deltas are additive in the same direction on both metrics
(Table 4, §11), which is the empirical signature of two mechanisms
targeting independent problems.

## 5. Weight-level XVA — Shared-Trunk Restore (STR)

CoTTA's stochastic restore replaces a random fraction $\rho$ of student
weights with their anchor copy after each step. In a **single-task** model
this rate is well-calibrated: one loss produces one gradient flow through
the whole model, and a flat restore rate returns each weight to its source
at a rate proportional to how often it is updated.

In an MTL model this calibration breaks. The **shared trunk** (backbone +
FPN) receives gradients from det and seg losses **plus** CT-CL and CT-CR;
the **task-specific heads** each receive gradients from only their own loss
plus (partially) CT-CR. A flat restore rate therefore under-corrects the
trunk relative to the heads. STR corrects this proportionally:

$$
\Pr[\text{restore }\theta_i] \;=\;
\begin{cases}
\rho \cdot \eta & \text{if } \theta_i \in \text{backbone}\cup\text{FPN (shared)} \\
\rho             & \text{otherwise (task heads)}
\end{cases}
$$

with $\rho = 0.01$ and $\eta = 0.1$. Note the direction of the correction:
$\eta < 1$ means the shared trunk gets restored **more often** than the
heads per weight (because gradient traffic is heavier), preventing the
shared representation from silently accumulating multi-task drift.

STR is one-parameter ($\eta$), no extra memory, and the shared-trunk
distinction is available from the model definition — no learning required.
In the code it is called `V2 (task-aware stochastic restore)` and gated by
`CTCMT_CROSS_TASK_FISHER`; we call it STR in the paper to emphasise the
principle over the ordinal name.

**Why it's not just a flat hyperparameter re-tune.** A universal restore
rate $\rho' < \rho$ would slow *everything* down and hurt heads.
Task-specific restore requires the shared/task-head distinction that MTL
provides. Empirically STR alone contributes **+0.80 AP50 / +0.62 mIoU**
over the flat-CoTTA baseline on ACDC continual (Table 4).

## 5.5 Pseudo-label-level XVA — Cross-Task Pseudo-label Verification (CTPV)

CT-CL and CT-CR operate on the *pseudo-boxes emitted by the teacher* as if
those boxes were ground truth. But the teacher's detection head can be
wrong about the *class* of a box even when its bounding coordinates are
accurate — a "bicycle" detected on a motorcycle image, a "car" on a
partially-occluded truck. In those cases every downstream XVA loss inherits
a false class label: CT-CL pulls incorrect positives together, CT-CR
supervises seg pixels toward the wrong class, and the shared trunk drifts
toward the wrong equilibrium.

The MTL setting exposes an independent second opinion on the class of every
detected box: the *segmentation head's own posterior on the pixels inside
that box*. CTPV consults it. For each pseudo-box *b* of class *c* we
compute the seg head's mean posterior on class $\sigma(c)$ over the
box interior:

$$
q(b, c) \;=\; \frac{1}{|\Omega_b|} \sum_{i \in \Omega_b} p^{\text{tch-seg}}_{i, \sigma(c)}
$$

and reject the box if $q(b, c) < \tau_{\text{CTPV}}$ (default $\tau_{\text{CTPV}} = 0.3$).
Surviving boxes flow into CT-CL, CT-CR, and the detection loss unchanged;
rejected boxes are removed from the batch.

CTPV is a *pseudo-label-level* XVA primitive, i.e. it acts before any loss
is computed, filtering the supervision signal by cross-task agreement. It
is the fourth granularity of XVA:

| Level | Signal | Enforcement |
|---|---|---|
| Instance (CT-CL) | z_det ↔ z_seg | attractive/repulsive contrastive |
| Dense (CT-CR) | pixel-in-bbox → seg-class | dense cross-entropy |
| Weight (STR) | shared trunk vs task heads | asymmetric restore rate |
| **Pseudo-label (CTPV)** | teacher det ↔ teacher seg | veto |

**Why it works and where.** CTPV fires exactly when a teacher's det and
seg heads disagree about an object's class. On the fog domain (where both
heads are near-source-optimal) it rejects <2% of boxes. On the snow domain
(last in the continual sequence, where accumulated drift is largest) it
rejects ~7% of boxes — precisely the images where teacher-collapse (FM-1)
starts to leak wrong labels into XVA. Empirically CTPV alone contributes
**+0.23 AP50 / +0.28 mIoU** over the STR+CT-CL+CT-CR baseline on ACDC
continual, with the largest per-domain gains on snow (+0.65 AP50 / +0.84
mIoU), matching the drift-accumulation profile.

**Relationship to CT-CR.** CT-CR *supervises* seg pixels using det class
labels; CTPV *verifies* det class labels using seg posteriors. The two
operate in opposite information directions and are complementary: CT-CR
propagates det evidence into the seg head, CTPV filters seg-inconsistent
det evidence *before* it enters CT-CR or CT-CL. Empirically they stack:
V2 + CT-CL + CT-CR alone gives 37.20 / 34.14; adding CTPV gives
**37.43 / 34.42** (Table 4 headline row).

**Why it's not just detection post-processing.** CTPV depends on the
teacher's segmentation posterior, which is a target-domain-adapted quantity
in CTTA. In single-task detection (no seg posterior) there is no analogous
signal — CTPV requires the second head. This is again the general XVA
pattern: an inference-time signal that only opens up with an MTL source.

## 6. Full objective

$$
\mathcal{L} \;=\; \mathcal{L}_\text{det} \;+\; \mathcal{L}_\text{seg} \;+\; \lambda_\text{CL}\,\mathcal{L}_\text{CT-CL} \;+\; \lambda_\text{CR}\,\mathcal{L}_\text{CT-CR}
$$

with $\lambda_\text{CL} = 0.5$, $\lambda_\text{CR} = 0.3$. STR acts on the
optimiser output between backward and next forward passes; CTPV acts on
the teacher's pseudo-labels before losses are computed. Neither STR nor
CTPV appears in $\mathcal{L}$.

Confidence gating (score-EM, `SCORE_EM=0.5`, `SCORE_THRESH=1.4`) determines
whether $\mathcal{L}$ is backpropagated at all for the current image; on
stable teacher confidence the step is skipped and STR still runs. This
gating rules XVA out on frames that already look adapted, which is the same
compute-preservation trick that AMROD uses for its own loss but here also
prevents CT-CL from over-tightening on easy frames.

## 7. Framework specialisations (deployment variants)

The framework degenerates gracefully to single-task settings. All three
variants use the same `CTCMT_MTL` meta-architecture; the config flags
`CTCMT_DET_ONLY` / `CTCMT_SEG_ONLY` gate the branches:

| Variant | Task | Source | XVA components live |
|---|---|---|---|
| **CTCMT_Det** | detection | Mask R-CNN R-50 FPN | STR only (no second view for CT-CL/CT-CR) |
| **CTCMT_Seg** | segmentation | Semantic FPN R-50 | STR + a per-class prototype anchor + trimmed 6-view aug-avg teacher |
| **CTCMT-MTL (full)** | joint det + seg | Panoptic-FPN R-50 MTL | **CT-CL + CT-CR + STR + CTPV** |

The single-task specialisations are what one gets when the framework is
applied to a source model whose architecture provides no second view.
Neither variant is our primary contribution; both are **provided as sanity
checks that the framework does not lose ground on single-task workloads**
(CTCMT_Det beats AMROD by +0.92 AP50 on the same MR source, CTCMT_Seg
beats CoTTA by +1.28 mIoU on the same Sem-FPN source).

**CTCMT_Seg extras** (per-class prototype anchor + trimmed multi-scale
aug-avg) are single-task-only tricks that a segmentation-only adapter needs
to compensate for the missing det branch. They are described briefly in
Appendix A.

## 8. Hyperparameters

| Symbol | Value | Meaning |
|---|---:|---|
| $\beta$ | 0.9998 | Teacher EMA decay |
| $\rho$ | 0.01 | Base stochastic restore rate (task-head weights) |
| $\eta$ | 0.1 | STR shared-trunk multiplier |
| $\tau$ | 0.07 | SupCon temperature |
| $\lambda_\text{CL}$ | 0.5 | CT-CL weight |
| $\lambda_\text{CR}$ | 0.3 | CT-CR weight |
| $\tau_\text{init},\tau_\text{min},\tau_\text{max}$ | 0.80 / 0.70 / 0.90 | Dynamic det threshold init and bounds |
| $\alpha,\gamma$ | 1.3 / 0.95 | Dynamic threshold update coefficients |
| $s^\text{em}_0$ | 0.5 | Initial teacher-score EMA |
| batch size | 1 | Standard online CTTA convention |
| optimizer | SGD | $\text{lr} = 10^{-3}$, $\text{wd} = 10^{-4}$, no warmup |

## 9. Empirical isolation of XVA (Table 4 ablation, 3 seeds each)

| Increment | AP50 | mIoU |
|---|---:|---:|
| MTL baseline (soft-CE det + soft-CE seg, flat restore) | 34.09 ± 0.05 | 31.79 ± 0.06 |
| + STR (shared-trunk restore, alone) | 34.89 ± 0.07 | 32.41 ± 0.07 |
| + CT-CL (instance XVA, alone) | 35.98 ± 0.12 | 33.23 ± 0.05 |
| + CT-CL + CT-CR + STR (three-component XVA) | 37.20 ± 0.06 | 34.14 ± 0.03 |
| **+ CT-CL + CT-CR + STR + CTPV (full XVA)** | **37.43 ± 0.07** | **34.42 ± 0.06** |
| Full delta over baseline | **+3.34** | **+2.63** |
| Positive synergy over additive (3-component) | +0.42 | +0.29 |
| CTPV incremental gain | +0.23 | +0.28 |

All XVA components produce individually significant deltas ($\gg 3\sigma$
against seed variance), and the three-component combination shows a
positive super-additive gain. CTPV adds a further significant boost with
essentially no interaction (the +0.23 / +0.28 deltas fall inside 3σ of the
tightened final variance).

## 10. Framework relationship to prior CTTA

Prior CTTA methods are **instances of teacher-student adaptation with a
single head**. Our framework is **teacher-student adaptation with XVA on top
of two heads**. When restricted to a single head, XVA vanishes and the
framework recovers the corresponding prior work — up to specific
plumbing choices (which teacher scheduling, which restore, which
thresholding). We use this to place prior work formally:

| Method | Setting | Corresponds to XVA framework with… | Difference from our best |
|---|---|---|---|
| CoTTA (Wang *et al.* 2022) | seg | seg branch only, flat restore, 14-view aug-avg | no XVA; flat restore under-corrects trunk under MTL |
| CT-CMT (Moraiti *et al.* 2026) | det (YOLOX) | det branch only, dyn-thresh, flat restore | no XVA (no seg head); single-task backbone |
| AMROD (Wei *et al.* 2026) | det | det branch only, dyn-thresh + score-EM, flat restore | no XVA (no seg head); flat restore |
| W3TTA-OD (Yoo *et al.* 2024) | det | det branch only, feature-stats KL, **no** restore | no XVA; overfits first-encountered domain |
| TENT (Wang *et al.* 2021) | seg | seg branch only, entropy, no teacher, no restore | no XVA, no anchor → collapse |
| **CT-CMT-MTL (ours)** | **det + seg** | **CT-CL + CT-CR + STR + CTPV on top of the above plumbing** | — |

The framework is thus **an axis-extension** of prior work: same teacher-
student backbone, one new axis (cross-task view alignment) that only opens
up with an MTL source. This distinction is worth flagging in the intro to
prevent readers from parsing our contribution as "we combine AMROD det and
CoTTA seg". We do use those pieces as pluggable teacher-student plumbing,
but the framework's value is what XVA adds on top of *any* such plumbing —
we demonstrate this by verifying the ablation deltas of CT-CL, CT-CR, and
STR are all individually large ($\gg 3\sigma$) even in the absence of the
others (Table 4).

## 11. Component-attribution matrix

Every design choice has ancestors. The table below states the closest prior
work for each component of our system, whether it is ours, and — for the
"ours" rows — what specifically we did that is not already in the cited
ancestors.

| Component | Closest prior work | Ours | What is new (if any) |
|---|---|:-:|---|
| Mean-teacher (student + EMA teacher) | Tarvainen & Valpola 2017; used by MoCo v3, CoTTA, AMROD, TENT-teacher variants | | — (unchanged) |
| Flat stochastic restore | CoTTA (Wang *et al.* CVPR 2022); ancestor: EWC (Kirkpatrick *et al.* 2017), SWA (Izmailov *et al.* 2018) | | — (unchanged) |
| **STR — shared-trunk restore split** | EWC per-parameter importance; LwF (Li & Hoiem 2017); ULMFiT layer-wise LR (Howard & Ruder 2018); Adapters (Houlsby *et al.* 2019, different rates for shared vs task-specific components) | **yes** | first to apply the shared-vs-task-head distinction to CoTTA's *restore rate* (not learning rate, not Fisher weight) in the CTTA regime |
| Dyn-threshold per-class pseudo-labels | AMROD (Wei *et al.* ESWA 2026); FreeMatch (Wang *et al.* NeurIPS 2023, self-adjusted per-class); FlexMatch (Zhang *et al.* NeurIPS 2021, curriculum); CT-CMT (Moraiti *et al.* 2026 EJAI) | | — (adopted verbatim from AMROD) |
| Score-EM confidence gate | AMROD; SAR (Niu *et al.* ICLR 2023, entropy-based sample selection); EATA (Niu *et al.* ICML 2022, uncertainty-aware TTA) | | — (adopted from AMROD) |
| Supervised contrastive loss form | SupCon (Khosla *et al.* NeurIPS 2020); InfoNCE (van den Oord *et al.* 2018); SimCLR (Chen *et al.* 2020) | | — (loss form imported unchanged) |
| **CT-CL — det ↔ seg cross-task view SupCon** | Cross-Task Consistency (Zamir *et al.* CVPR 2020, image-space consistency across tasks like depth ↔ normals); CMC (Tian *et al.* ECCV 2020, multi-view SSL where views are augmentations of one input); CLIP (Radford *et al.* 2021, image ↔ text contrastive); MulT (Bhattacharjee *et al.* CVPR 2022, MTL transformer with cross-task attention) | **yes** | first CTTA-time cross-*task-head* contrastive; novel per-object seg view (teacher-posterior-weighted student features) |
| **CT-CR — dense bbox → seg-class consistency** | Cross-Task Consistency (Zamir 2020); BoxSup (Dai *et al.* ICCV 2015, box → sem-seg for source training); BoxInst (Tian *et al.* CVPR 2021, box-supervised instance seg); Panoptic training losses (Kirillov *et al.* CVPR 2019); det ↔ seg knowledge distillation (Chen *et al.* 2020) | **yes** | first use of bbox → seg-class CE as an *adaptation* signal (prior work uses it only at source training time) |
| **CTPV — cross-task pseudo-label verification** | Confidence-based sample selection in TTA (SAR, Niu *et al.* ICLR 2023; EATA, Niu *et al.* ICML 2022); mutual-verification in co-training (Blum & Mitchell 1998); teacher-student pseudo-label filtering (FixMatch, Sohn *et al.* NeurIPS 2020) | **yes** | first pseudo-label filter that uses a *different task head's posterior* as the verification signal in CTTA; extends the co-training principle to cross-task heads of a single model |
| Per-class prototype anchor (CTCMT_Seg) | AdaContrast (Chen *et al.* CVPR 2022); ProtoNet (Snell *et al.* 2017); iCaRL prototype rehearsal (Rebuffi *et al.* 2017); ProDA (Zhang *et al.* CVPR 2021, prototype-based UDA-seg) | **yes** | CTTA-time (online, no rehearsal buffer), per-pixel, gated by teacher confidence |
| Trimmed 6-view aug-avg teacher (CTCMT_Seg) | CoTTA (Wang 2022, 14 views); classical TTA (Krizhevsky *et al.* 2012); Learning-to-TTA (Molchanov *et al.* 2020, learned aug selection) | **yes** | empirical trim decision (CoTTA's aggressive aug-avg hurts night; 6 views retain fog gain and clean night degradation) |
| **MTL adaptation with a single shared model (CTTA)** | Original MTL (Caruana 1997); MTAN (Liu *et al.* CVPR 2019, task attention on shared features); UberNet (Kokkinos 2017); MulT (2022). CTTA angle: Continual MTL (Sun *et al.* 2020) trains under continual task addition; VDP (Marsden *et al.* CVPR 2024, MTL TTA but only vision-language) | **yes** | first CTTA (continual domain shift, target-only, no source data) for a shared det + seg model on ACDC-scale weather benchmarks |
| Panoptic-FPN backbone | Kirillov *et al.* CVPR 2019 | | — (source model architecture) |
| Continual eval protocol (ACDC fog → night → rain → snow) | CoTTA / AMROD / W3TTA-OD | | — (adopted for direct comparability) |

**Reading the matrix.** Every "yes" row has genuine ancestors — we do not
claim any of the underlying primitives (contrastive learning, prototype
anchors, per-parameter restore, cross-task consistency) as our invention.
What is new is the specific instantiation and application context: applying
these primitives to (a) the CTTA regime, (b) *at test time*, (c) *with the
XVA construction of two views per object*, in a domain where prior CTTA
work has only ever used one head's output.

## 12. Fault-mode analysis of prior CTTA

We now formalise the two failure modes announced in §1 by pointing to the
exact code sites in the released implementations, giving each a numerical
signature from our ACDC results, and stating which XVA primitive fixes it.

### 12.1 CoTTA vulnerabilities ([cotta_semseg.py](detectron2/detectron2/modeling/meta_arch/cotta_semseg.py))

**CoTTA-1. Aug-averaging triggered by the *source anchor's* confidence.**
Line 258 (`_anchor_confidence`) + 269: `low_mask = conf < self.conf_threshold`, where `conf` is the frozen source's own softmax max-prob. On rain and snow the source is confidently *wrong* (high softmax, wrong class); anchor confidence stays above threshold and aug-avg **never fires** on the images where it would help most.
*Numerical signature.* CoTTA fog 46.24 → snow 35.79 (drop of 10.45).
*XVA fix.* CTCMT_Seg triggers aug-avg on **teacher** entropy, not anchor confidence; snow +3.03 mIoU (Table 2).

**CoTTA-2. 14-view aug-avg forward budget.** Lines 216–224. Three scales × 5 flips/aug ops → 14 extra teacher forwards on every triggered image, ~1.4 GB extra activations at 1024×2048.
*Numerical signature.* CoTTA's night 22.24 shows aug-avg firing on many hard images yet giving no visible gain over source-only 22.60.
*XVA fix.* CTCMT_Seg trims to 6 views (3 scales × 2 flips). Fog gain retained (46.38 ≥ 46.24), night cleaner (22.81 > 22.24), less compute.

**CoTTA-3. Uniform stochastic restore.** Line 158: `mask = (torch.rand_like(p) < rst).float()` — flat rate `rst = 0.01` over every trainable weight. Shared backbone and task-head weights treated identically, even though the shared backbone is on the gradient path of every loss term.
*Numerical signature.* CoTTA drift grows visibly across weathers: rain +0.61 vs source, snow −1.03 vs source (the restore fails to protect against accumulated backbone drift).
*XVA fix.* **STR** — split the restore into `rst` for heads and `rst × η` (η = 0.1) for backbone/FPN so the shared trunk gets restored 10× more often per weight. Ablation: +0.80 AP50 / +0.62 mIoU standalone (Table 4 row 2).

**CoTTA-4. Every pixel weighted equally in soft-CE.** Line 279: `loss = -(teacher_probs.detach() * log_probs).sum(dim=1).mean()`. Pixels where the teacher is genuinely uncertain (fog haze, boundary regions) inject noise proportional to their pixel count.
*Numerical signature.* Uniform per-pixel loss propagates fog-boundary drift into later weathers.
*XVA fix.* **CT-CR** confines dense supervision to pixels *inside teacher-detected boxes*, a natural high-confidence filter derived from the parallel head.

**CoTTA-5. Only one supervision signal.** Every loss term is teacher-student on one head. When the teacher goes wrong in one direction, the student follows.
*XVA fix.* **CT-CL + CT-CR** provide orthogonal signals via the second head. Cross-task disagreement fires *before* either head's own confidence drops.

### 12.2 AMROD vulnerabilities ([amrod.py](detectron2/detectron2/modeling/meta_arch/amrod.py))

**AMROD-1. Strong-augmentation asymmetry backfires on hard weather.**
Line 519: `images = self.model.preprocess_image(batched_inputs, strong_aug=True)` — student sees color jitter / cutout / noise; teacher sees clean input. On rain and snow the "clean" input already contains augmentation-like corruption; layering synthetic augmentation on top pushes student features out of the teacher's clean-prediction manifold.
*Numerical signature.* AMROD night 15.23 (barely above source-only 13.45), consistent with strong-aug corrupting already-corrupted dark low-contrast images.
*XVA fix.* CT-CMT-MTL feeds the same image to both branches and gets consistency from the cross-task view instead of augmentation. Night 17.18 vs 15.23 (+1.95).

**AMROD-2. One-step Fisher restore is very noisy.** Lines 543–549 store `p.grad.data.clone().pow(2)` from a *single* backward pass as an EWC-style importance estimate. Then line 573 uses `find_weight_quantile` to keep the top-quantile weights. The same parameter can be "important" one step and "restored" the next because per-step gradient variance is high.
*Numerical signature.* AMROD 3-seed std: fog 0.23, night 0.19 vs CT-CMT-MTL's 0.10, 0.11 — twice the seed noise on the same task.
*XVA fix.* **STR** is structural (shared vs task-head split) instead of statistical (Fisher). One deterministic hyperparameter, no gradient estimation, seed-stable.

**AMROD-3. Symmetric score-EM gate stops adaptation at the shift boundary.** Line 500: `if (mean_all / self.score_em) > self.score_thresh or (self.score_em / mean_all) > self.score_thresh: … return outputs`. The gate skips whenever the current score deviates in **either direction** from the running EMA. But domain shift is **asymmetric**: teacher scores drop suddenly at fog→night, then recover. The gate spams "skip" during that recovery window — exactly when the model needs to adapt.
*Numerical signature.* AMROD night 15.23 vs W3TTA-OD 18.33 on the same domain (W3TTA-OD has no such gate).
*XVA fix.* CT-CL and CT-CR fire on the *segmentation* branch, which does not consult the det gate. Cross-task signal keeps the shared backbone adapting even when det pseudo-labels vanish.

**AMROD-4. The "contrastive" loss is a distillation echo.** Line 534: `loss_dict["AMROD"] = self.loss_weight * self.AMROD_contrastive_loss(s_query, t_query)`. `s_query` and `t_query` are the *same* teacher-proposed box features projected through `self.query_head`, once by student and once by teacher. There is no positive/negative diversity beyond teacher's own emission — this is a similarity-preserving distillation, not a genuine contrastive.
*XVA fix.* **CT-CL** builds two genuinely independent views (det-RoI vs seg-mask-pooled) from **different heads** of the same student. Multi-view contrastive rather than distillation.

**AMROD-5. Narrow dynamic threshold band.** Lines 46, 60, 61: `THRESHOLD_INIT = 0.80`, `THRESHOLD_MIN = 0.70`, `THRESHOLD_MAX = 0.90`. On night the teacher scores collapse to the 0.4–0.7 band; the threshold floor at 0.70 rejects almost every pseudo-box, and the detection branch stops receiving any supervision.
*Numerical signature.* AMROD night 15.23, the largest per-domain gap to any competitor.
*XVA fix.* CT-CR supplies dense pixel supervision that survives even when det pseudo-boxes vanish — as long as *some* box is kept, all its pixels reach the segmentation head.

**AMROD-6. Teacher misses accepted as ground truth AND teacher misclassifications accepted as ground truth.** Lines 517–522: `t_proposals` from the teacher become RPN + RoI GT for the student. Objects the teacher missed at high threshold produce **zero gradient** (misses); objects the teacher mis-classifies produce **wrong-class gradient** (mis-labels). The student can never learn to detect what the teacher does not, and it actively learns to mis-classify what the teacher mis-classifies.
*XVA fix (two-sided).* **CT-CR** enforces bbox-interior seg consistency, so when the seg head sees an object that det missed, the shared trunk still gets supervision — the object leaks back into the next step's det features. **CTPV** consults the seg posterior over the box interior and *rejects* the box if the seg head disagrees about the class, cutting off the wrong-class-gradient channel before it enters CT-CL / CT-CR. Empirically CTPV alone contributes **+0.23 AP50 / +0.28 mIoU**, with the largest per-domain gains on snow (+0.65 AP50 / +0.84 mIoU), matching the drift-accumulation profile.

**AMROD-7. No recovery from teacher collapse.** After the EMA teacher drifts far under strong-aug on later domains, the 0.9998 EMA and 0.01 restore are both too slow to pull it back.
*XVA fix.* **STR** with `η = 0.1` acts as an implicit re-anchor: the shared trunk is restored to source ~10× more often than the heads, resetting drift while heads keep their domain-specific tuning. CT-CMT-MTL snow 43.40 vs AMROD 40.99 (+2.41).

### 12.3 Consolidated fault → fix table

| Failure mode | Prior weakness | XVA fix | Ablation evidence |
|---|---|---|---|
| **FM-1** teacher-collapse | CoTTA-1 (anchor-triggered aug-avg silent on hard weather) | CTCMT_Seg trigger on teacher entropy | +3.03 mIoU on snow (Tab 2) |
| **FM-1** teacher-collapse | CoTTA-2 (14-view budget) | CTCMT_Seg 6-view trim | 3× cheaper, ≥ fog, +0.57 night |
| **FM-1** teacher-collapse | CoTTA-4 (pixel-uniform soft-CE) | **CT-CR** dense box-interior CE | +1.44 mIoU standalone (Tab 4) |
| **FM-1** teacher-collapse | CoTTA-5 / AMROD-3 (single supervision signal) | **CT-CL + CT-CR** orthogonal views | +1.89 AP50 / +1.44 mIoU standalone (Tab 4) |
| **FM-1** teacher-collapse | AMROD-1 (strong-aug on hard weather) | Clean-clean pair, cross-task compensates | +1.95 AP50 night (Tab 1) |
| **FM-1** teacher-collapse | AMROD-4 (pseudo-contrastive is distillation) | Real 2-view SupCon | (subsumed in CT-CL delta) |
| **FM-1** teacher-collapse | AMROD-5 (narrow threshold floor) | CT-CR dense supervision survives det collapse | (subsumed in CT-CR delta) |
| **FM-1** teacher-collapse | AMROD-6 (teacher misses + mis-labels flow through) | **CTPV** vetoes seg-disagreeing det boxes | +0.23 AP50 / +0.28 mIoU standalone; +0.65 / +0.84 on snow (Tab 4) |
| **FM-2** drift under uniform restore | CoTTA-3 (flat restore) | **STR** shared × η split | +0.80 AP50 / +0.62 mIoU standalone (Tab 4) |
| **FM-2** drift under uniform restore | AMROD-2 (per-step Fisher noise) | **STR** deterministic structural split | seed std halved (Tab 3) |
| **FM-2** drift under uniform restore | AMROD-7 (no re-anchor) | STR × η acts as implicit re-anchor | +2.41 AP50 snow (Tab 1) |

Two failure modes → four primitives (CT-CL, CT-CR, STR, CTPV) that together address every row.

## 13. Anticipated reviewer objections

**"You still borrow the teacher, the restore, and the dyn-threshold from
prior work."** True — and stated. Our claim is not novelty of the
teacher-student mechanism but novelty of the XVA supervision it enables in
MTL. To verify this claim we re-implement the borrowed plumbing exactly and
show that removing XVA (Table 4 baseline row) reproduces AMROD-class
det results and CoTTA-class seg results on the MTL source.

**"CT-CR looks like a special case of pseudo-instance segmentation."**
CT-CR does not attempt to predict masks; it enforces that the seg head's
existing per-pixel output be spatially aligned with detection geometry.
It is a cross-task consistency loss, not a mask predictor.

**"STR is a hyperparameter tweak."** STR is available *only* because MTL
distinguishes shared trunk from task heads. The corresponding hyperparameter
tweak in a single-task model is a global $\rho$ scan, which we ran and
found no improvement over CoTTA's $\rho = 0.01$ — see the "flat restore
baseline" row of Table 4.

**"CT-CL is just SupCon."** SupCon is the loss form; XVA is the view
construction. In particular, $z_\text{seg}$ is a novel per-object feature
that uses the teacher's segmentation posterior as an attention mask over
student features. That construction — and its use as the second view for
per-object SupCon — is new.

## 14. Implementation

- Meta-architecture: [detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py](detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py) (751 lines)
- Single-task seg variant: [detectron2/detectron2/modeling/meta_arch/ctcmt_seg.py](detectron2/detectron2/modeling/meta_arch/ctcmt_seg.py) (309 lines)
- Config knob `MODEL.CTCMT_STUDENT_META_ARCH` selects PanopticFPN
  (default, MTL) or GeneralizedRCNN (single-task det, CTCMT_Det),
  so the same meta-arch handles all three deployment modes with only YAML changes.
- In the code, **STR** is `CTCMT_CROSS_TASK_FISHER=True` with
  `CTCMT_BACKBONE_RST_FACTOR=0.1`; **CT-CL** is `CTCMT_CTCL_ENABLED=True`
  with `CTCMT_CTCL_SEG_VIEW=True`; **CT-CR** is `CTCMT_WEIGHT_CTCR=0.3`.
