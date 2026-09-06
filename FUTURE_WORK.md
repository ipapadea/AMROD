# Future Experimental Work — Roadmap

Prioritised list of experiments still needed to bring the paper to a strong
IEEE / Springer / top-CV venue submission. Each item includes the target
gap it closes, GPU-time estimate, and a priority score (★★★★★ = must-have).

Ranking is based on: (a) reviewer expectations at CVPR / TPAMI / IJCV /
Pattern Recognition, (b) direct continuity with the prior Moraiti *et al.*
EJAI 2026 paper, and (c) marginal value per GPU-hour.

---

## Tier 1 — Must-have to be defensible

### 1.1. Reset-of-source ("loopback") evaluation ★★★★★
- **What**: after the fog → night → rain → snow ACDC sequence, evaluate
  the adapted model on the original Cityscapes val set. Report
  `Forgetting = mAP_Source − mAP_Loopback`.
- **Why**: identical to the "Forgetting" metric used throughout Moraiti
  *et al.* 2026 as the central catastrophic-forgetting evidence. Reviewers
  will look for this line.
- **Effort**: 3 seeds × ~15 min per run for MTL, plus 3 seeds × ~15 min for
  each single-task variant. **~2 GPU-hours total.**
- **Status**: pending — [scripts/tmux_loopback.sh](scripts/tmux_loopback.sh)
  (to be added).

### 1.2. Second benchmark beyond ACDC ★★★★★
- **What**: run the full XVA framework on at least one more continual
  benchmark. Two natural choices:
  - **SHIFT** (Sun *et al.* CVPR 2022) — the benchmark from the EJAI 2026
    prior paper. Strongest continuity story.
  - **Cityscapes-C or KITTI-C** — corruption benchmarks (noise / blur /
    weather).
- **Why**: single-benchmark results are fragile. Prior EJAI paper used 5
  datasets; this paper currently uses ACDC + Foggy CS.
- **Effort**: SHIFT ≈ 12 GPU-hours (6 sequences × 4 seeds × ~30 min);
  Cityscapes-C ≈ 6 GPU-hours (15 corruption types × 3 seeds × ~10 min).

### 1.3. Runtime & memory efficiency table ★★★★★
- **What**: wall-clock per-image latency (ms) and peak GPU memory (GB) for
  each competitor and each of our variants.
- **Why**: CTTA is billed as an online real-time technique. Reviewers ask
  for cost.
- **Deliverable**: table of the form
  | Method | ms/img | Peak GPU (GB) | Rel. cost |
- **Effort**: latency can be extracted from existing per-iter logs (free);
  peak memory needs a fresh nvidia-smi run per method (~1 GPU-hour).

### 1.4. Hyperparameter sensitivity ★★★★☆
- **What**: sweep the four new hyperparameters (η, τ_CTPV, λ_CL, λ_CR)
  each over 3-5 values, 1 seed each.
- **Why**: shows the method is robust to reasonable HP choices, not tuned
  per-experiment.
- **Effort**: 4 sweeps × 4 values × 1 seed = ~16 runs × ~15 min = ~4
  GPU-hours.

## Tier 2 — Should have for a strong paper

### 2.1. Additional competitor baselines ★★★★☆
- **What**: add 2-3 recent CTTA methods for direct comparison.
  - **RMT** (Döbler *et al.* CVPR 2023) — Robust Mean Teacher, seg
  - **AR-TTA** (Sójka *et al.* ICCVW 2023) — memory-buffer CTTA
  - **ActMAD** (Mirza *et al.* CVPR 2023) — activation matching
  - **BeCoTTA** (Lee *et al.* ICLR 2024) — mixture-of-experts CTTA
  - **SVDP** (Wang *et al.* AAAI 2024) — sparse visual domain prompts
- **Why**: Table 1 currently has 3 competitors (Source, W3TTA-OD, AMROD).
  Reviewers expect 4-5 total on a modern benchmark.
- **Effort**: each method ≈ 1-2 days to port/adapt + 3 seeds × 15 min.
  **~4-6 GPU-days for 3 additional methods.**

### 2.2. Domain-order shuffling ★★★★☆
- **What**: report results on 2-3 alternative continual orderings.
  - `night → snow → rain → fog` (reverse difficulty)
  - `snow → fog → rain → night` (random)
  - `fog → rain → night → snow` (mixed)
- **Why**: pre-empt "you gamed the fog→night→rain→snow order" reviewer
  concern.
- **Effort**: 3 orderings × 3 seeds × ~15 min = ~2.5 GPU-hours.

### 2.3. Continual seg comparison with recent methods ★★★☆☆
- **What**: add SVDP + BeCoTTA to the Table 2 (single-task seg) row.
- **Effort**: ~2 GPU-days for the two additional methods.

### 2.4. Per-class analysis ★★★☆☆
- **What**: per-class AP / IoU for the 8 det + 19 seg classes on ACDC snow
  (the worst-drift domain).
- **Why**: standard in detection papers; opens the "XVA helps rare classes"
  argument.
- **Effort**: extractable from existing logs; **~1 hour of scripting**.

## Tier 3 — Optional strengtheners

### 3.1. t-SNE / UMAP of learned embeddings ★★★☆☆
- **What**: visualise `z_det` and `z_seg` embeddings before/after XVA,
  coloured by class. Show they collapse to the same per-class code.
- **Why**: strong visual evidence of the paper's central claim; excellent
  Figure candidate.
- **Effort**: 1 day scripting + 1 seed of adaptation with feature-dumping
  enabled.

### 3.2. Long-sequence stability ★★★☆☆
- **What**: run ACDC 2× or 3× back-to-back
  (fog→night→rain→snow→fog→night→…) and check for drift.
- **Effort**: 3 seeds × ~45 min per triple-run = ~2 GPU-hours.

### 3.3. Alternative backbone generality ★★☆☆☆
- **What**: repeat CTCMT-MTL headline with a Swin-Tiny or ConvNeXt-T
  backbone.
- **Why**: pre-empts "does XVA generalise beyond R-50?" reviewer concern.
- **Effort**: ~2 days to train a new PFN source + 3 seeds of adaptation.
  **~1 GPU-week.**

### 3.4. Ablation on projection heads ★★☆☆☆
- **What**: compare Conv+MLP vs plain MLP heads; embedding dim 64/128/256;
  with/without L2 normalisation.
- **Effort**: ~3 GPU-hours total.

### 3.5. Statistical significance testing ★★☆☆☆
- **What**: paired t-tests / Wilcoxon signed-rank between our method and
  each baseline. P-values in a supplementary table.
- **Effort**: ~2 hours of scripting.

## Non-experiment gaps

### 4.1. Architecture diagram (Fig. 1) ★★★★★
- Vector figure showing student / teacher / anchor, the 4 XVA loss paths,
  and the EMA + STR loop. Currently only an ASCII sketch exists in
  METHOD.md.

### 4.2. XVA views figure (Fig. 2) ★★★★☆
- Replicate the style of Fig. 2 in Moraiti *et al.* 2026 (arrows between
  teacher and student RoIs) but with two views per bbox instead of one.

### 4.3. Ablation bar chart (Fig. 3) ★★★☆☆
- Bar chart of the 4-primitive incremental gains (baseline → +STR → +CT-CL
  → +CT-CR → +CTPV) with error bars.

### 4.4. Related-work section ★★★★★
- Currently minimal. Needs a proper CTTA landscape survey with at least
  30 references organised into: source-free UDA, TTA, continual TTA,
  cross-task learning, MTL adaptation.

---

## Recommended next-batch prioritisation

If you have **1 GPU-day**: 1.1 (loopback) + 1.3 (runtime table) + 2.4
(per-class analysis).

If you have **1 GPU-week**: everything in Tier 1 + Cityscapes-C in 1.2 +
one Tier 2 competitor (RMT is the cheapest port from mmsegmentation).

If you have **1 month**: everything in Tier 1 + all of Tier 2 + t-SNE +
alternative backbone. Puts the paper at "no-brainer accept" for a strong
venue.
