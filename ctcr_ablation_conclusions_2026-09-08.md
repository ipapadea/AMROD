# CT-CR Ablation Conclusions and Next Steps

_Date: 2026-09-08_

## Scope

This note summarizes the current conclusions from the CT-CR ablation study for the multi-task continual test-time adaptation setup.

The three compared CT-CR formulations are:

- **A — Full-box CT-CR**
  - Legacy formulation.
  - Supervises the entire detection bounding box with the mapped semantic class.
  - Uses the legacy target-map construction, including overwrite behavior in overlapping boxes.

- **B — Hard semantic-support CT-CR**
  - Per-box loss.
  - Uses detached teacher semantic probability for the mapped detector class.
  - Supervises only pixels satisfying `q(i) >= 0.3`.

- **C — Soft semantic-support CT-CR**
  - Per-box weighted loss.
  - Supervises the whole box, but weights each pixel using:
    \[
    w(i)=\alpha+(1-\alpha)q(i)
    \]
  - Current setting: `alpha = 0.2`.

All A/B/C experiments discussed here use:

- MTL model
- CT-CL enabled
- V2 task-aware/backbone-protected stochastic restoration
- CTPV disabled
- same overall continual adaptation framework

---

# 1. Main 3-seed results

## Cityscapes-C mixed long-term

Protocol:

\[
(fog \rightarrow motion\_blur \rightarrow snow \rightarrow brightness \rightarrow defocus\_blur)\times 10
\]

No reset between domains or rounds.

| CT-CR | AP50 mean ± std | mIoU mean ± std |
|---|---:|---:|
| **A — full box** | 18.723 ± 0.456 | 29.100 ± 0.064 |
| B — hard semantic support | 18.457 ± 0.717 | 29.265 ± 0.191 |
| **C — soft semantic support** | **18.851 ± 0.746** | **29.357 ± 0.268** |

### Interpretation

C has the highest mean on the mixed long-term benchmark, but the improvement over A is small:

\[
C-A = +0.128\ AP50
\]

\[
C-A = +0.257\ mIoU
\]

The AP50 difference is much smaller than the observed seed-to-seed variation, therefore there is currently **no strong evidence that C is clearly better than A for detection** in the mixed long-term setting.

The semantic result is more consistent: C is above A in mean mIoU and is above A in each of the three mixed-stream seeds.

---

# 2. Seed-wise behavior on Cityscapes-C mixed long-term

## AP50

| Seed | A | B | C |
|---|---:|---:|---:|
| 0 | **18.9588** | 17.7088 | 18.0060 |
| 42 | 18.1980 | 18.5235 | **19.1260** |
| 123 | 19.0124 | 19.1385 | **19.4198** |

### Key observation

C consistently outperforms B:

- seed 0: `C - B = +0.2972 AP50`
- seed 42: `C - B = +0.6025 AP50`
- seed 123: `C - B = +0.2813 AP50`

This makes the hard-mask formulation B difficult to justify as the final CT-CR design.

However, C versus A is not stable across seeds:

- seed 0: `C - A = -0.9528`
- seed 42: `C - A = +0.9280`
- seed 123: `C - A = +0.4074`

Therefore, A versus C remains unresolved for detection on the mixed stream.

---

# 3. ACDC 3-seed results

Protocol:

\[
fog \rightarrow night \rightarrow rain \rightarrow snow
\]

No reset between domains.

| CT-CR | AP50 mean ± std | mIoU mean ± std |
|---|---:|---:|
| **A — full box** | **37.343 ± 0.047** | **34.206 ± 0.108** |
| B — hard semantic support | 36.863 ± 0.120 | 33.964 ± 0.095 |
| C — soft semantic support | 36.678 ± 0.136 | 33.754 ± 0.163 |

### Differences relative to A

For B:

\[
A-B = +0.479\ AP50
\]

\[
A-B = +0.242\ mIoU
\]

For C:

\[
A-C = +0.665\ AP50
\]

\[
A-C = +0.453\ mIoU
\]

### Main conclusion

On ACDC, A is clearly the strongest formulation overall.

Unlike the mixed Cityscapes-C stream, the ACDC result is not ambiguous:

- A wins in average AP50.
- A wins in average mIoU.
- A wins consistently across all three seeds.

---

# 4. ACDC domain-wise behavior

Three-seed AP50 means:

| Domain | A | B | C |
|---|---:|---:|---:|
| fog | 52.708 | 52.766 | **52.940** |
| night | 16.943 | 16.926 | **16.966** |
| rain | **35.765** | 35.145 | 34.852 |
| snow | **43.954** | 42.617 | 41.953 |

### Important pattern

At the beginning of the stream, C is competitive:

- fog: `C - A = +0.232 AP50`
- night: `C - A = +0.023 AP50`

As the continual stream progresses, C degrades relative to A:

- rain: `C - A = -0.913 AP50`
- snow: `C - A = -2.001 AP50`

This suggests that the disadvantage of soft semantic modulation is not simply an initial-domain effect. It appears to **accumulate during continual domain switching**.

A similar trend appears in semantic segmentation.

Approximate mIoU differences relative to A:

| Domain | B - A | C - A |
|---|---:|---:|
| fog | -0.212 | -0.257 |
| night | -0.025 | -0.117 |
| rain | -0.354 | -0.660 |
| snow | -0.376 | -0.776 |

Thus, on ACDC, A eventually dominates both tasks rather than simply trading detection performance for segmentation performance.

---

# 5. Long-term behavior on Cityscapes-C

Approximate three-seed round-level behavior:

| Variant | R1 AP50 | Peak AP50 | R10 AP50 |
|---|---:|---:|---:|
| A | 16.373 | ~19.69 | 18.034 |
| B | 16.032 | ~19.51 | 17.617 |
| C | 16.344 | ~19.85 | **18.358** |

C has the strongest final detection performance among the three.

For segmentation:

| Variant | R1 mIoU | Peak mIoU | R10 mIoU |
|---|---:|---:|---:|
| A | 30.638 | 32.106 | 25.777 |
| B | 29.834 | 31.502 | **26.524** |
| C | 30.270 | 32.086 | 26.326 |

Peak-to-last semantic decay:

- A: `-6.329 ± 0.679`
- B: `-4.989 ± 0.639`
- C: `-5.760 ± 0.332`

### Interpretation

B shows the best semantic retention in the long-term mixed stream, despite being weaker overall.

This suggests that hard semantic support may have a regularizing or protective effect on the segmentation branch.

However, this protection is not sufficient to make B the best overall method because it loses more in detection and does not perform best on ACDC.

---

# 6. Fog ×10 diagnostic

Single-seed controlled repeated-fog experiment:

| Variant | AP50 mean | mIoU mean |
|---|---:|---:|
| A | 41.0037 | 43.7876 |
| B | 41.8192 | 43.8318 |
| **C** | **43.1046** | **45.2351** |

### Interpretation

C is clearly strongest when the target domain is repeated and stable.

This creates an important contrast with ACDC:

- **stable repeated domain:** soft semantic weighting is beneficial
- **continually changing domains:** the benefit weakens or reverses

A plausible hypothesis is that teacher semantic confidence `q(i)` is a useful spatial reliability estimate after adaptation within a stable domain, but can become temporarily stale or unreliable immediately after domain transitions.

This is currently an interpretation of the results, not yet a proven mechanism.

---

# 7. Current view of the three CT-CR formulations

## A — Full-box

Strengths:

- strongest on ACDC
- very stable across seeds
- strongest or highly competitive on mixed long-term
- simple and robust

Weaknesses:

- known full-box label noise
- person/rider/bicycle-like classes can have poor semantic purity
- overlapping boxes use legacy overwrite behavior
- not spatially selective

Current status:

**Still the strongest overall reference formulation.**

---

## B — Hard semantic support

Strengths:

- improves over full-box in the repeated-fog diagnostic
- best semantic retention in the long-term mixed stream
- explicitly suppresses low-semantic-confidence pixels

Weaknesses:

- worse detection than C in all three mixed-stream seeds
- worse than A on ACDC
- hard threshold discards supervision
- threshold `tau=0.3` is class-dependent and not universally optimal

Current status:

**Unlikely to be the final CT-CR formulation.**

---

## C — Soft semantic support

Strengths:

- best on repeated fog
- best mean AP50 and mIoU on mixed long-term
- consistently better than B
- avoids hard discard of low-confidence pixels
- better semantic behavior than A on mixed long-term

Weaknesses:

- not consistently better than A in mixed-stream AP50
- worse than A on ACDC
- performance gap versus A grows later in the ACDC stream
- possible sensitivity to domain transitions

Current status:

**Most promising semantic-aware CT-CR formulation, but not yet clearly superior to the legacy full-box formulation.**

---

# 8. Important experimental confound: A vs B/C aggregation

The comparison is not yet perfectly controlled.

A uses:

- legacy full target map
- full-box semantic assignment
- overlap overwrite behavior

B and C use:

- per-detection box losses
- per-box normalization
- averaging over boxes
- no target-map overwrite behavior

Therefore, if B or C changes performance, the difference may come from either:

1. semantic spatial masking/weighting, or
2. the change from global target-map aggregation to per-box aggregation

These factors must be separated.

---

# 9. Required next CT-CR control: A2

Define:

\[
A2 = \text{per-box full CT-CR}
\]

A2 should use:

- the full bounding-box rectangle
- no semantic threshold
- no semantic weighting
- the same per-box loss computation as B/C
- the same per-box normalization as B/C
- mean across detections

Then:

- `A vs A2` measures the effect of aggregation / overlap behavior
- `A2 vs B` measures the effect of hard semantic support
- `A2 vs C` measures the effect of soft semantic weighting

This is the most important remaining control before making a strong claim about the CT-CR redesign.

---

# 10. Hardware confound to correct

Current Cityscapes-C mixed seeds were not all run on the same GPU architecture.

Current layout:

- seed 0:
  - A/B/C on Cronus / RTX 3090
- seed 42:
  - A on Cronus / RTX 3090
  - B/C on gpu1 / L40S
- seed 123:
  - A/B/C on gpu1 / L40S

Even with TF32 disabled, online CTTA trajectories can be sensitive to low-level numerical and hardware differences.

Recommended correction:

Run:

- **A full-box**
- Cityscapes-C mixed long-term ×10
- seed 42
- gpu1 / L40S

This gives:

- seed 0: A/B/C all on RTX 3090
- seed 42: A/B/C all on L40S
- seed 123: A/B/C all on L40S

It also provides a direct hardware reproducibility comparison for the same A seed42 run.

---

# 11. Recommended next steps for the paper

## Immediate

1. Run the hardware-correction experiment:
   - A mixed LT seed42 on L40S.

2. Implement A2:
   - per-box full CT-CR.

3. Screen A2:
   - ACDC seed0
   - Cityscapes-C mixed LT seed0

4. If A2 is meaningfully different:
   - repeat seeds 42 and 123.

5. Lock the CT-CR formulation.

---

## After CT-CR is locked

6. Revisit CTPV / cross-task reliability.

The current CTPV is a hard whole-detection veto based on semantic agreement.

Diagnostics showed that it rejects many correct detections, therefore the next design should probably use **reliability-aware coupling rather than binary rejection**.

7. Add proper source-retention evaluation.

Important: the clean Cityscapes loopback must be evaluated in a **no-update mode**.

Otherwise, evaluating on clean Cityscapes while continuing adaptation would partially re-adapt the model and hide source forgetting.

8. Build the final paper ablation:

\[
MTL
\rightarrow +CTCL
\rightarrow +CTCR
\rightarrow +V2
\rightarrow +\text{cross-task reliability}
\]

9. Final multi-seed evaluation:

- ACDC
- Cityscapes-C
- long-term streams
- clean Cityscapes loopback
- detection AP/AP50
- semantic mIoU
- forgetting / retention metrics
- runtime and memory overhead

---

# 12. Current high-level conclusion

The current evidence does **not** support a simple claim that semantic masking is always better than full-box CT-CR.

Instead:

> Semantic spatial reliability is useful under stable target-domain adaptation, but its benefit becomes less consistent under continual domain transitions.

Hard masking appears too aggressive.

Soft semantic weighting is more promising than hard masking, but the legacy full-box formulation remains the strongest overall baseline, especially on ACDC.

The next decisive experiment is the **A2 per-box full-box control**, because it will separate the effect of semantic spatial reliability from the effect of changing the loss aggregation mechanism.

Only after that control should the final CT-CR formulation be selected.
