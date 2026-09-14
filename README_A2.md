# A2 per-box full CT-CR control

A2 keeps full-box supervision but switches from the legacy global target map
to the same per-box normalization/averaging used by B/C.

Install on each machine:

```bash
cd ~/AMROD
unzip -o a2_perbox_full_ctcr_bundle.zip -d .
python3 scripts/patch_a2_perbox_full.py
chmod +x scripts/*a2*.sh
bash scripts/smoke_a2_perbox_full.sh
```

Expected: `A2 UNIT TEST PASSED`.

Recommended runs:
- Cronus GPU0: A2 mixed seed0
- gpu1 GPU0: A2 mixed seed42
- gpu1 GPU1: A2 mixed seed123
- gpu1 GPU2: A2 ACDC seeds 0,42,123, then A2 fogx10 seed0
- gpu1 GPU3: use for legacy A seed42 L40S hardware correction
