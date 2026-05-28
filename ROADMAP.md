# Roadmap

The point of this file is to keep me honest. Anyone reading the repo
can tell what was promised and whether it shipped.

## 30 days (by 2026-06-27)

- [x] Public scaffold (this commit)
- [ ] Kvasir-SEG loader + manifest schema with 100% test coverage
- [ ] Pretrained YOLOv8n-seg baseline running end-to-end against
      the Kvasir-SEG public test split
- [ ] First eval report committed to `reports/` with:
  - Polyp Dice
  - Per-frame sensitivity + specificity
  - Per-clip lead time (synthetic — proprietary clips not yet IRB-cleared)
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
