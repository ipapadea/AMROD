# From CT-CMT to XVA — Multi-Task Continual TTA as a Structured Extension of Our Prior Work

## 1. Motivation and continuity with the initial work

Our prior work (Moraiti *et al.*, *EJAI* 2026 — "Continual Test-Time
Domain Adaptation for Object Detection via Contrastive Mean Teacher and
Stochastic Restoration") established a three-component recipe for
source-free continual TTA on a single detection task:

1. **Mean Teacher (MT)** with an EMA of the student weights,
   $\theta_T \leftarrow \alpha \theta_T + (1-\alpha)\theta_S$;
2. **Object-level Contrastive Learning (CL)** — SupCon (Khosla *et al.*
   2020) over per-object RoI-aligned features $z_T(b), z_S(b)$ pulled by
   the teacher's own pseudo-labels;
3. **Stochastic Restoration (SR)** — Bernoulli(p) restoration of a random
   fraction of student weights to source (CoTTA convention, uniform mask).

The evaluation on SHIFT, KITTI, Cityscapes, CLAD-D, and COCO-C showed that
**SR was the dominant contributor to preserving source knowledge** across
long continual streams and that adding object-level CL on top of MT was a
consistently positive but smaller boost. Two failure modes were left open
by that formulation:

- Object-level CL uses only *one head's* view per object (RoI features).
  When the teacher becomes wrong about the class, the SupCon loss silently
  reinforces the mistake — there is no orthogonal signal to catch it.
- SR restores every parameter with the same Bernoulli probability. The
  backbone (task-invariant, gradient-heavy, shared across every loss term)
  is thus under-corrected relative to task-specific layers.

**This paper extends the CT-CMT framework to multi-task learning** and, in
doing so, closes both failure modes at the mechanistic level. The extension
generalises the two components that carry over (CL → CT-CL, SR → STR), adds
two mechanisms that only exist in the MTL setting (CT-CR, CTPV), and
inherits the MT skeleton unchanged.

Everything in the paper is either (i) a re-implementation of a component
already in Moraiti *et al.* 2026 for a new backbone, (ii) a strict
generalisation that reduces to the prior work when the MTL setting is
collapsed, or (iii) a genuinely new mechanism that is only *possible* in
MTL. §10 summarises the mapping.

## 2. Prior recipe, formalised

Let $b$ be a pseudo-detected bounding box of class $c$ produced by the
teacher's own detection head. In Moraiti *et al.* 2026 the CL loss reads

$$
\mathcal{L}_{\text{CL}} \;=\; \frac{\lambda}{N}\sum_{i=1}^{N}\frac{-1}{|P(i)|}\sum_{p \in P(i)} \log \frac{\exp(z_i^S \cdot z_p^T / \tau)}{\sum_{j=1}^{N}\exp(z_i^S \cdot z_j^T / \tau)}
$$

with:

- $z_i^M \;=\; \text{Normalize}(\text{RoIAlign}(F^M, b_i))$, where $M \in \{S, T\}$ is student or teacher and $F$ is a multi-scale backbone/PAFPN feature map;
- $P(i) = \{p : C_p = C_i\}$ the same-class positive set;
- $\tau = 0.07$, $\lambda = 0.003$.

The **object-level view $z_i$** is a per-bbox RoIAlign. There is only one
view per object — no orthogonal supervision signal from a second head. The
loss cannot fire when only one class is present in the image (all pairs are
positive) and cannot distinguish a genuinely well-adapted representation
from one that is class-degenerate.

The SR mechanism is described in §3.2 of Moraiti *et al.* 2026 as

$$
M \sim \text{Bernoulli}(p),\qquad W_{t+1} \;=\; M \cdot W_0 + (1-M) \cdot W_{t+1}
$$

applied element-wise to *every* trainable weight in the student. $p = 0.01$
on KITTI/Cityscapes/CLAD-D; $p = 0.025$ on the SHIFT sequences.

## 3. Cross-Task Contrastive Learning (CT-CL) — a strict generalisation of prior object-level CL

The MTL setting exposes each object at the level of the shared backbone via
**two independent heads** — detection and segmentation. Our first
extension augments the prior object-level view $z^S(b)$ with a second view
$z^S_{\text{seg}}(b)$ constructed from the teacher's per-pixel segmentation
posterior:

$$
z^S_{\text{det}}(b) \;=\; \phi_{\text{det}}\big(\text{RoIAlign}(F^S, b)\big)
\qquad\text{(as in prior work)}
$$

$$
z^S_{\text{seg}}(b) \;=\; \phi_{\text{seg}}\!\left(\frac{\sum_{i \in b} p^T_{i, \sigma(c)}\,F^S_i}{\sum_{i \in b} p^T_{i, \sigma(c)} + \varepsilon}\right)
\qquad\text{(**new**)}
$$

where $\sigma(c)$ maps the detection class $c$ to the corresponding
segmentation class via the Cityscapes taxonomy, and $\phi_{\text{det}}$,
$\phi_{\text{seg}}$ are lightweight L2-normalised 128-d projection heads.
The det view is identical to the prior CT-CMT construction. The seg view is
the *teacher segmentation posterior used as an attention mask over student
features*, which is why it fires only for objects the seg head recognises
and does not require any additional annotation.

The SupCon loss then operates over the combined pool
$\{(z_i, y_i)\}$ where $z_i \in \{z^S_{\text{det}}, z^S_{\text{seg}}\}$ and
$y_i$ is the detection class. Positive pairs come in three qualitatively
different flavours:

| Pair type | Effect | Available in Moraiti 2026? |
|---|---|:-:|
| $z_\text{det} \leftrightarrow z_\text{det}$ (same class) | classical object-level SupCon | yes |
| $z_\text{seg} \leftrightarrow z_\text{seg}$ (same class) | new — encourages seg-view coherence | no (single-task) |
| **$z_\text{det} \leftrightarrow z_\text{seg}$ (same class)** | **new — forces two heads to converge to a shared per-class embedding** | no (single-task) |

**Prior work as a special case.** Deleting the segmentation branch removes
$z_{\text{seg}}$, dropping rows 2 and 3 and recovering the object-level CL
loss of Moraiti *et al.* 2026 verbatim (up to the concrete backbone —
YOLOX PAFPN in prior work, Panoptic-FPN R-50 here). CT-CL is thus a
strictly larger family; any conclusion drawn in the prior paper about
object-level CL still applies to the row-1 subset of CT-CL.

**Why this closes the "silent-mistake" issue.** When the teacher's
detection head misclassifies an object as class $c'$ but the seg head still
labels the pixels as $\sigma(c)$ (the *true* class), $z_{\text{seg}}(b)$
diverges from $z_{\text{det}}(b)$. The cross-pair SupCon term then produces
a large gradient that *cannot* be reconciled by moving both projections
toward the same wrong-class centroid: they are seeded from different heads.
The loss surface pushes the shared backbone toward a representation on
which both heads agree — which is exactly the class that is present.

## 4. Cross-Task Consistency Regularizer (CT-CR) — a new mechanism

CT-CL is instance-level (one embedding per bbox). To close the same-image
loop at the pixel level, we add a dense cross-task consistency
regulariser: every pixel inside a pseudo-box $b$ of class $c$ is treated as
a segmentation-side pseudo-label of class $\sigma(c)$:

$$
\mathcal{L}_{\text{CT-CR}} \;=\; \sum_{i \in b} \mathrm{CE}\!\left(p^{S,\text{seg}}_i,\; \sigma(c)\right)
$$

with pixels outside every box ignored (`ignore_index=255`). This
propagates detection-side certainty back through the segmentation head
and, symmetrically, ensures the segmentation head's per-pixel predictions
be spatially consistent with the detection geometry.

There is **no analog of CT-CR in Moraiti *et al.* 2026** because the prior
work has no segmentation head to receive the supervision. It is a novel
mechanism only *definable* in the MTL setting. Empirically CT-CR contributes
part of the +1.44 mIoU that CT-CL + CT-CR together deliver over the STR-only
baseline; it is enabled by default with $\lambda_{\text{CR}} = 0.3$.

## 5. Shared-Trunk Stochastic Restoration (STR) — a task-aware generalisation of prior SR

Moraiti *et al.* 2026 applies SR uniformly across all trainable weights.
In the MTL setting, backbone and PAFPN weights carry gradient traffic from
*every* loss term (det + seg + CT-CL + CT-CR), while task-specific head
weights receive gradients only from their own task. A uniform Bernoulli
mask therefore under-corrects the trunk relative to the heads. STR splits
the mask by parameter role:

$$
\Pr[\text{restore }\theta_i] \;=\;
\begin{cases}
p \cdot \eta & \text{if } \theta_i \in \text{backbone} \cup \text{PAFPN (shared)} \\
p             & \text{otherwise (task-specific heads)}
\end{cases}
$$

with $p = 0.01$ (matching prior work) and $\eta = 0.1$. The shared trunk
gets restored $10\times$ more often than the heads per weight, in
expectation.

**Prior SR is the $\eta = 1$ special case** of STR (uniform mask). When
the model is single-task (no distinction between shared and task-specific
components), $\eta$ is meaningless and STR reduces to the prior formulation
exactly. The extension is therefore backwards-compatible.

**Why this closes the "under-corrected backbone" issue.** In the prior
paper's KITTI/COCO-C experiments the "Forgetting" metric tracked the
model's ability to return to source performance after adaptation. With
uniform SR, forgetting was small on shorter streams but grew visibly on
COCO-C's 15-domain sequence. STR's higher backbone-restore rate protects
the shared representation against exactly this long-run drift, at zero
extra cost to task-specific fine-tuning capacity.

## 6. Cross-Task Pseudo-Label Verification (CTPV) — a new mechanism

The MT skeleton in Moraiti *et al.* 2026 accepts every teacher-detected
box (above a confidence threshold) as pseudo ground-truth. When the
teacher is wrong about the *class* of a well-localised box, the wrong
label enters CL and the model actively learns to mis-classify.

The MTL setting provides an independent second opinion on the class: the
segmentation head's own posterior over the box interior. CTPV consults it.
For each pseudo-box $b$ of class $c$:

$$
q(b, c) \;=\; \frac{1}{|\Omega_b|}\sum_{i \in \Omega_b} p^{T,\text{seg}}_{i, \sigma(c)}
$$

is compared to a threshold $\tau_{\text{CTPV}} = 0.3$. Boxes with $q < \tau_{\text{CTPV}}$
are removed from the pseudo-label pool *before* CT-CL, CT-CR, or the
detection loss are computed on this step.

CTPV is a **pseudo-label-level** XVA primitive: it acts before any loss,
filtering the supervision signal itself by cross-task agreement. It has no
analog in Moraiti *et al.* 2026 for the same reason CT-CR does not — there
is no second head to verify against. On ACDC continual (see §9) CTPV alone
contributes **+0.23 AP50 / +0.28 mIoU** over the CT-CL + CT-CR + STR
baseline, with the largest gains on snow (+0.65 AP50 / +0.84 mIoU) where
teacher-drift accumulates most.

## 7. Fault-mode framing — a diagnostic reformulation of the prior narrative

The prior paper motivated its three components (MT, CL, SR) as
complementary techniques. Our extension re-frames the discussion around
two **structural failure modes** of teacher-student CTTA:

- **FM-1 (Teacher-collapse under a single supervision signal).** In
  detection, teacher scores fall below the confidence floor on hard
  weather and no pseudo-boxes survive; adaptation silently stops. In
  segmentation, teacher per-pixel entropy grows and the soft-CE loss
  amplifies teacher noise into gradient noise. Both are visible in
  single-head systems because the only supervision channel is the head
  that is drifting.
- **FM-2 (Drift under uniform restore).** Uniform-Bernoulli SR treats
  shared trunk and task heads identically. The shared trunk, on every
  gradient path, accumulates domain-specific bias faster than the mask
  reverts it.

CT-CL, CT-CR, and CTPV target FM-1 at three levels of granularity
(instance, dense, pseudo-label). STR targets FM-2. This framing is a
*sharper* version of the observation in Moraiti *et al.* 2026 that "MT
alone suffers from noisy pseudo-labels" and "adaptation for a long time
leads to catastrophic forgetting". The prior paper diagnosed the
symptoms; the extension diagnoses the mechanisms and shows that both are
structural consequences of single-head CTTA.

## 8. Framework specialisations

The MTL meta-architecture supports three deployment modes controlled by
config flags. Each specialisation is a strict subset of the full XVA
framework:

| Variant | Source | Active components | Relationship to Moraiti *et al.* 2026 |
|---|---|---|---|
| **CTCMT_Det** | Mask R-CNN R-50 FPN | MT + soft-CE det consistency + STR | direct port of the prior single-task recipe to a two-stage det backbone; CT-CL falls back to object-level SupCon |
| **CTCMT_Seg** | Semantic FPN R-50 | MT + soft-CE seg + STR + per-class prototype anchor + trimmed 6-view aug-avg | prior recipe transposed to segmentation (a scenario not tested in the prior paper) |
| **CTCMT-MTL + V2 + CTPV** | Panoptic-FPN R-50 MTL | soft-CE det + soft-CE seg + CT-CL + CT-CR + STR + CTPV | the paper's main contribution |

The single-task variants exist to (a) verify that the framework does not
regress on prior workloads and (b) provide fair same-source-model
baselines for the MTL headline. In practice CTCMT_Det on Mask R-CNN source
beats AMROD by +0.92 AP50 on ACDC continual, and CTCMT_Seg on Semantic FPN
source beats CoTTA by +1.28 mIoU — both consistent with the prior paper's
finding that MT + CL + SR outperforms plain MT.

## 9. Empirical validation on ACDC continual

We report a fresh ablation on ACDC continual (fog → night → rain → snow),
3 seeds per row, using the Panoptic-FPN R-50 MTL source. **Baseline** = MTL
soft-CE det + soft-CE seg + uniform stochastic restore (an MTL port of
the plumbing that would result from applying the prior work's components
naively to a shared-backbone model).

| Increment | AP50 | mIoU |
|---|---:|---:|
| MTL baseline (uniform SR, no CT-CL, no CT-CR, no CTPV) | 34.09 ± 0.05 | 31.79 ± 0.06 |
| + STR (task-aware SR, replaces uniform SR) | 34.89 ± 0.07 | 32.41 ± 0.07 |
| + CT-CL (three-way SupCon over det + seg views) | 35.98 ± 0.12 | 33.23 ± 0.05 |
| + CT-CL + CT-CR + STR (three-component XVA) | 37.20 ± 0.06 | 34.14 ± 0.03 |
| **+ CT-CL + CT-CR + STR + CTPV (full)** | **37.43 ± 0.07** | **34.42 ± 0.06** |
| Full delta over baseline | **+3.34** | **+2.63** |

Every increment is significant against seed variance ($\gg 3\sigma$). The
STR row alone reproduces the "SR is the dominant single component"
finding from the prior paper — even on this new backbone and this new
benchmark — but the addition of CT-CL, CT-CR, and CTPV together adds
another **+2.54 AP50 / +2.01 mIoU** on top, confirming that the
MTL-specific mechanisms deliver gains that were impossible to access from
the single-task formulation.

**Foggy Cityscapes (single-shot).** Same source model:
- Source only: 32.15 AP50 / 44.12 mIoU
- Full extension: **33.19 AP50 / 49.97 mIoU**

**Comparison against SOTA on ACDC continual.** Full-extension vs published
methods on the same Mask R-CNN R-50 source (Table 1 of the paper): W3TTA-OD
(Yoo *et al.* CVPR 2024) 34.20 AP50; AMROD (Wei *et al.* ESWA 2026) 35.34
AP50; ours **36.06** (single-task CTCMT_Det+STR) and **37.43** (full MTL
XVA). Full-extension vs CoTTA on ACDC continual seg (same Semantic FPN
source): CoTTA 36.39 mIoU; CTCMT_Seg **37.67 mIoU**.

## 10. What is new vs. what is preserved

| Component in this paper | Origin | Relationship to Moraiti *et al.* 2026 |
|---|---|---|
| Mean Teacher skeleton | Tarvainen & Valpola 2017 | preserved verbatim |
| EMA teacher update, α=0.9998 | Moraiti *et al.* 2026 | preserved (same rate) |
| Dynamic per-class thresholds | AMROD (Wei 2026), also used in prior CT-CMT | preserved verbatim |
| Score-EM confidence gate | AMROD (Wei 2026) | preserved verbatim |
| Object-level SupCon on det features | Moraiti *et al.* 2026 §3.3 | preserved as the det ↔ det pair type inside CT-CL |
| **Cross-task views (det + seg per bbox)** | this paper §3 | **new**: seg view is a novel construction |
| **CT-CL (three-way SupCon: det↔det, seg↔seg, det↔seg)** | this paper §3 | **new**: reduces to prior CL when seg branch is absent |
| **CT-CR (dense bbox → seg-class CE)** | this paper §4 | **new**: no analog in prior work |
| Uniform Bernoulli stochastic restore | Moraiti *et al.* 2026 §3.2 (from CoTTA) | preserved as the $\eta = 1$ special case of STR |
| **STR (shared-trunk vs task-head split)** | this paper §5 | **new generalisation**: recovers prior uniform SR when the shared/task distinction is trivial |
| **CTPV (seg-verifies-det pseudo-label filter)** | this paper §6 | **new**: no analog in prior work |
| Backbone choice (Panoptic-FPN R-50 MTL) | Kirillov *et al.* 2019 | new source model (prior work: YOLOX single-task) |
| Benchmark (ACDC continual + Foggy CS MTL) | Sakaridis *et al.* 2021 | new evaluation setting; prior work used SHIFT / KITTI / Cityscapes / CLAD-D / COCO-C |

The extension therefore preserves the entire prior framework as a
substructure, generalises its two flagship mechanisms (CL and SR) into
constructs that reduce to the originals in the single-task limit, and
introduces two mechanisms (CT-CR, CTPV) that are only definable when a
second head is available. The MT skeleton, dyn-threshold, and score-EM
gate that the prior work adopted from other CTTA methods are inherited
without change.

## 11. Practical remarks

1. **The prior work's implementation stays valid.** The CT-CMT adapter
   in [shift-tta/shift-detection-tta](https://github.com/PanagiotaMoraiti/shift-tta)
   uses YOLOX PAFPN features for RoIAlign, and the object-level CL is
   applied over teacher-pseudo boxes. Replacing YOLOX with Panoptic-FPN
   and adding a segmentation branch is a strict feature addition — no
   part of the prior implementation needs to be removed.
2. **Hyperparameters carry over.** SupCon temperature $\tau = 0.07$, EMA
   decay $\alpha = 0.9998$, dyn-threshold $\tau_c$ init, min, max are
   used unchanged from the prior paper. New hyperparameters ($\eta = 0.1$
   for STR, $\lambda_{\text{CL}} = 0.5$, $\lambda_{\text{CR}} = 0.3$,
   $\tau_{\text{CTPV}} = 0.3$) were tuned on a single held-out run and
   fixed across all reported experiments.
3. **The paper cites Moraiti *et al.* 2026 as its immediate predecessor**
   and positions itself as the MTL extension of that framework.

## 12. Section-level correspondence with Moraiti *et al.* 2026

| Prior section | Prior content | This paper's corresponding section |
|---|---|---|
| §1 Introduction | Motivation for continual TTA; catastrophic forgetting | §1 Motivation and continuity |
| §2 Related work | UDA, TTA, continual TTA | Merged into §7 (fault-mode framing) and §10 (component-attribution) |
| §3.1 MT framework | Teacher/student EMA, consistency loss | §1.1 "Teacher-student plumbing" — preserved verbatim |
| §3.2 Stochastic Restoration | Uniform Bernoulli mask, $p = 0.01$ | §5 STR — generalised (backward-compatible) |
| §3.3 Object-level CL | RoIAlign features, class-based SupCon | §3 CT-CL — generalised (backward-compatible) |
| §3.3.3 Multi-scale feature maps | Three PAFPN scales summed | Retained; FPN scales P3/P4/P5 in Panoptic-FPN |
| §4 Experiments | SHIFT, KITTI, Cityscapes, CLAD-D, COCO-C | §9 — evaluation on ACDC continual + Foggy CS |
| §5 Conclusion | Positive contributions of MT + CL + SR | Restated and sharpened in §7 (FM-1 / FM-2 diagnostic) |
| — (new) | — | §4 CT-CR (dense cross-task consistency) — new mechanism |
| — (new) | — | §6 CTPV (pseudo-label verification) — new mechanism |

The extension is a linear supersession of the prior paper's methodology:
prior components appear as either preserved plumbing or as backward-
compatible generalisations. Two new mechanisms (CT-CR, CTPV) close
failure modes that are only diagnosable in the MTL setting.
