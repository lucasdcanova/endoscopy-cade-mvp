# Roadmap

The point of this file is to keep me honest. Anyone reading the repo
can tell what was promised and whether it shipped.

## 30 days (by 2026-06-27)

- [x] Public scaffold (this commit)
- [x] Kvasir-SEG loader + manifest schema with test coverage (`src/cade/data/kvasir_seg.py` + `tests/test_manifest.py`)
- [~] Pretrained YOLOv8n-seg baseline pipeline scripted (`scripts/train_kvasir_baseline.py` + `scripts/eval_kvasir_baseline.py`). Awaiting GPU run; download script at `scripts/download_kvasir_seg.sh`.
- [~] First eval report layout finalised (`reports/<run>.json` + `<run>.md`). Metric layer ships Dice + IoU + sens + spec + balanced accuracy + p50/p95 latency. Awaiting first real GPU run.
- [ ] `notebooks/01_eda_kvasir.ipynb` — exploratory data analysis
- [ ] Single-page write-up (`docs/baseline-report.md`) explaining the
      result honestly: what worked, what didn't, what the next experiment is

## 60 days (by 2026-07-27)

- [ ] MedSAM-2 baseline running on the same Kvasir-SEG split
- [ ] Side-by-side comparison report (YOLOv8 vs MedSAM-2) on the same
      manifest with the same seed
- [ ] IRB protocol drafted for the proprietary clip archive at
      Clínica Cirúrgica Canova — see `DATASET_GOVERNANCE.md`
- [ ] Paris classification annotation schema added to `src/cade/data/manifest.py`
- [ ] First 50 proprietary frames anonymised + annotated + scored
      (under IRB) — eval report committed without raw imagery

## 90 days (by 2026-08-27)

- [ ] Live demo: pretrained baseline running inside the endoscopy
      suite on a recorded clip, on-CPU, < 200 ms / frame
- [ ] Eval write-up sufficient for a MICCAI 2026 / EndoVis workshop
      submission
- [ ] Decision point: continue solo or open up the first co-founding /
      cohort conversation

## Beyond 90 days

Intentionally vague. The point of the 30/60/90 plan is to make decisions
in public; pretending to know what month 7 looks like would be theatre.
