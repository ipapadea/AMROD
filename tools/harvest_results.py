#!/usr/bin/env python3
"""Harvest all CTTA experiment results into results/registry.json.

Scans known log locations (detectron2 d2 + TriLiteNet), extracts metrics,
and writes a structured JSON registry. Run at any time — idempotent.

Usage:
    python3 tools/harvest_results.py            # update registry
    python3 tools/harvest_results.py --print    # pretty-print current registry
    python3 tools/harvest_results.py --print --filter acdc   # filter by tag
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REGISTRY_FILE = RESULTS_DIR / "registry.json"

# ---------------------------------------------------------------------------
# D2 log scanner
# ---------------------------------------------------------------------------

D2_CORRUPTIONS = [
    "defocus_blur","glass_blur","motion_blur","zoom_blur",
    "snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression",
]
ACDC_WEATHERS = ["acdc_fog","acdc_night","acdc_rain","acdc_snow"]

def _parse_d2_log(logfile: Path) -> Optional[Dict]:
    """Parse a detectron2 CTTA log → dict of {dataset: ap50}."""
    if not logfile.is_file():
        return None
    text = logfile.read_text(errors="replace")
    if "CTTA_" not in text and "Evaluation results for" not in text:
        return None

    results = {}
    # Find all (dataset_name, copypaste_line) pairs
    for m in re.finditer(
        r"Evaluation results for (\S+) in csv format.*?copypaste: ([0-9.,]+)",
        text, re.DOTALL
    ):
        ds, cp = m.group(1), m.group(2)
        vals = [float(v) for v in cp.split(",")]
        if len(vals) >= 2:
            # copypaste format is already percentage (e.g. 52.04); normalize to fraction
            results[ds] = {"AP": vals[0]/100, "AP50": vals[1]/100}

    return results if results else None


def _d2_summary(results: Dict) -> Dict:
    """Compute mean AP50 across all datasets in a d2 result dict."""
    ap50s = [v["AP50"] for v in results.values() if "AP50" in v]
    return {"mean_AP50": round(sum(ap50s)/len(ap50s), 4) if ap50s else None,
            "n_datasets": len(results)}


# ---------------------------------------------------------------------------
# TriLiteNet log scanner
# ---------------------------------------------------------------------------

def _parse_trilit_summary(summary_json: Path) -> Optional[Dict]:
    """Parse a TriLiteNet adapt_ctta summary.json."""
    if not summary_json.is_file():
        return None
    try:
        d = json.loads(summary_json.read_text())
    except Exception:
        return None
    out = {
        "global_miou": d.get("global", {}).get("miou"),
        "global_AP50": d.get("global", {}).get("AP50"),
        "per_weather": {},
        "per_repeat": [],
    }
    for w, v in d.get("per_weather", {}).items():
        out["per_weather"][w] = {"miou": v.get("miou"), "AP50": v.get("AP50")}
    for r in d.get("per_repeat", []):
        out["per_repeat"].append({
            "miou": r.get("miou"),
            "det": {w: v.get("AP50") for w, v in r.get("det", {}).items()},
        })
    return out


# ---------------------------------------------------------------------------
# Known experiment registry
# ---------------------------------------------------------------------------

D2_OUTPUT_ROOT = Path("/data/ilias/panoptic_fpn/output/ctta_acdc")
CS_C_ROOT = Path("/data/ilias/panoptic_fpn/output/ctta_cs_c")
TRILIT_CTTA_ROOT = Path("/home/ilias/TriLiteNet/runs/ctta")

D2_EXPERIMENTS = [
    # tag, output_subdir, benchmark, description
    # subdir is relative to D2_OUTPUT_ROOT for all entries (CS-C also lands in ctta_acdc/)
    ("amrod_acdc",          "amrod",               "ACDC",    "AMROD, MkRCNN src, Cityscapes→ACDC"),
    ("ctcmt_det_acdc",      "ctcmt_det",           "ACDC",    "CT-CMT-Det, PFN src, Cityscapes→ACDC"),
    ("ctcmt_seg_acdc",      "ctcmt_seg",           "ACDC",    "CT-CMT-Seg, PFN src, Cityscapes→ACDC"),
    ("ctcmt_mtl_acdc",      "ctcmt_mtl",           "ACDC",    "CT-CMT-MTL v4b, PFN src, Cityscapes→ACDC"),
    ("amrod_x10",           "amrod_x10",           "ACDC-10", "AMROD long-term 10-round"),
    ("ctcmt_det_x10",       "ctcmt_det_x10",       "ACDC-10", "CT-CMT-Det long-term 10-round"),
    ("ctcmt_v4b_x10",       "ctcmt_v4b_x10",       "ACDC-10", "CT-CMT-MTL v4b long-term 10-round"),
    ("amrod_shift",         "amrod_shift",         "SHIFT",   "AMROD, Cityscapes→SHIFT"),
    ("ctcmt_det_shift",     "ctcmt_det_shift",     "SHIFT",   "CT-CMT-Det, Cityscapes→SHIFT"),
    # Cityscapes-C (output lands in ctta_acdc/<track_name>/ due to OUT_ROOT override)
    ("amrod_cs_c",          "amrod_cs_c",          "CS-C",    "AMROD, MkRCNN-ours src, CS→CS-C"),
    ("amrod_official_cs_c", "amrod_official_cs_c", "CS-C",    "AMROD, official src, CS→CS-C"),
    ("ctcmt_det_cs_c",      "ctcmt_det_cs_c",      "CS-C",    "CT-CMT-Det, PFN src, CS→CS-C"),
    ("ctcmt_det_mr_cs_c",   "ctcmt_det_mr_cs_c",   "CS-C",    "CT-CMT-Det, MkRCNN src, CS→CS-C"),
    ("ctcmt_seg_cs_c",      "ctcmt_seg_cs_c",      "CS-C",    "CT-CMT-Seg, PFN src, CS→CS-C"),
    ("ctcmt_mtl_cs_c",      "ctcmt_mtl_cs_c",      "CS-C",    "CT-CMT-MTL v4b, PFN src, CS→CS-C"),
    ("cotta_cs_c",          "cotta_cs_c",          "CS-C",    "CoTTA-Seg, PFN src, CS→CS-C"),
]

TRILIT_EXPERIMENTS = [
    # tag, run_name, benchmark, description
    # --- MTL source ---
    ("trilit_source_acdc",      "cityscapes_mtl_base_native__source__x1",     "ACDC",    "TriLiteNet MTL, source-only"),
    ("trilit_amrod_acdc",       "cityscapes_mtl_base_native__amrod_mtl__x1",  "ACDC",    "TriLiteNet MTL, AMROD-MTL adapter"),
    ("trilit_ctcmt_acdc",       "cityscapes_mtl_base_native__ctcmt__x1",      "ACDC",    "TriLiteNet MTL, CT-CMT adapter"),
    ("trilit_ctcmt_x10",        "cityscapes_mtl_base_native__ctcmt__x10",     "ACDC-10", "TriLiteNet MTL, CT-CMT long-term"),
    ("trilit_ctcmt_det_acdc",   "cityscapes_mtl_base_native__ctcmt_det__x1",  "ACDC",    "TriLiteNet MTL, CT-CMT det-only"),
    ("trilit_ctcmt_seg_acdc",   "cityscapes_mtl_base_native__ctcmt_seg__x1",  "ACDC",    "TriLiteNet MTL, CT-CMT seg-only"),
    ("trilit_ctcmt_mtl_acdc",   "cityscapes_mtl_base_native__ctcmt_mtl__x1",  "ACDC",    "TriLiteNet MTL, CT-CMT MTL-only"),
    # --- Det-only source ---
    ("trilit_det_source_acdc",  "cityscapes_det_base_native__source__x1",     "ACDC",    "TriLiteNet det-only, source-only"),
    ("trilit_det_amrod_acdc",   "cityscapes_det_base_native__amrod__x1",      "ACDC",    "TriLiteNet det-only, AMROD adapter"),
    ("trilit_det_ctcmt_acdc",   "cityscapes_det_base_native__ctcmt__x1",      "ACDC",    "TriLiteNet det-only, CT-CMT adapter"),
    ("trilit_det_amrod_x10",    "cityscapes_det_base_native__amrod__x10",     "ACDC-10", "TriLiteNet det-only, AMROD long-term"),
    # --- Seg-only source ---
    ("trilit_seg_source_acdc",  "cityscapes_seg_base_native__source__x1",     "ACDC",    "TriLiteNet seg-only, source-only"),
    ("trilit_seg_cotta_acdc",   "cityscapes_seg_base_native__cotta__x1",      "ACDC",    "TriLiteNet seg-only, CoTTA adapter"),
    ("trilit_seg_ctcmt_acdc",   "cityscapes_seg_base_native__ctcmt__x1",      "ACDC",    "TriLiteNet seg-only, CT-CMT adapter"),
    ("trilit_seg_cotta_x10",    "cityscapes_seg_base_native__cotta__x10",     "ACDC-10", "TriLiteNet seg-only, CoTTA long-term"),
]


# ---------------------------------------------------------------------------
# Main harvester
# ---------------------------------------------------------------------------

def harvest() -> Dict:
    entries = {}

    # D2 experiments
    for tag, subdir, bench, desc in D2_EXPERIMENTS:
        logf = D2_OUTPUT_ROOT / subdir / "log.txt"
        parsed = _parse_d2_log(logf)
        if parsed:
            entries[tag] = {
                "tag": tag,
                "benchmark": bench,
                "description": desc,
                "framework": "detectron2",
                "log": str(logf),
                "summary": _d2_summary(parsed),
                "per_dataset": parsed,
                "harvested_at": datetime.utcnow().isoformat(),
            }

    # TriLiteNet experiments
    for tag, run_name, bench, desc in TRILIT_EXPERIMENTS:
        summary_f = TRILIT_CTTA_ROOT / run_name / "summary.json"
        parsed = _parse_trilit_summary(summary_f)
        if parsed:
            entries[tag] = {
                "tag": tag,
                "benchmark": bench,
                "description": desc,
                "framework": "trilitenet",
                "summary_json": str(summary_f),
                "metrics": parsed,
                "harvested_at": datetime.utcnow().isoformat(),
            }

    return entries


def pretty_print(registry: Dict, filter_str: Optional[str] = None) -> None:
    rows = sorted(registry.values(), key=lambda x: (x["benchmark"], x["tag"]))
    if filter_str:
        rows = [r for r in rows if filter_str.lower() in json.dumps(r).lower()]

    bench = None
    for r in rows:
        if r["benchmark"] != bench:
            bench = r["benchmark"]
            print(f"\n=== {bench} ===")
        tag = r["tag"]
        desc = r["description"]
        s = r.get("summary") or r.get("metrics", {})
        mean_ap50 = s.get("mean_AP50") or s.get("global_AP50")
        miou = s.get("global_miou")
        metrics = ""
        if mean_ap50 is not None:
            metrics += f"  AP50={mean_ap50*100:.1f}%"
        if miou is not None:
            metrics += f"  mIoU={miou*100:.1f}%"
        print(f"  {tag:<35} {desc:<45}{metrics}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="Print registry to stdout")
    ap.add_argument("--filter", default=None, help="Filter by substring")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REGISTRY_FILE.read_text()) if REGISTRY_FILE.is_file() else {}

    harvested = harvest()
    existing.update(harvested)

    REGISTRY_FILE.write_text(json.dumps(existing, indent=2))
    print(f"[harvest] {len(harvested)} entries harvested → {REGISTRY_FILE}")
    print(f"[harvest] Registry total: {len(existing)} entries")

    if args.print:
        pretty_print(existing, args.filter)


if __name__ == "__main__":
    main()
