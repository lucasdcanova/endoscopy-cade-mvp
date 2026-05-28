# endoscopy-cade-mvp

**Computer-aided detection (CADe) baseline + reproducible eval harness for
upper-GI endoscopy and colonoscopy — built by a practising endoscopist
to be the substrate of a clinician-led startup, not a research demo.**

> Status: bootstrapping (2026-05-28). Public benchmark on **Kvasir-SEG**
> first, NBI-NICE/Paris-classified proprietary clip evaluation second
> (under IRB), product surface third. See `ROADMAP.md`.

## Why this exists

There are three credible, ambient-scribe-style AI products in healthcare
right now (Abridge, Ambience, OpenEvidence) and roughly zero credible
**procedural** AI products that ship into the endoscopy suite end-to-end
with the endoscopist in the loop. The technical gap is small. The clinical
distribution gap is the moat — and it is held by the practising
endoscopist who can:

- Collect anonymised footage at scale from a clinic they own
- Annotate it with Paris / NICE / NBI taxonomy without paying an
  annotation vendor
- Run a real CADe model alongside a real Olympus CV-180 NBI tower in a
  real procedure
- Co-author the eval methodology that regulators will eventually require

This repository is the open, reproducible half of that. The closed half
(the proprietary frame archive, the annotation tooling, the clinical
trial protocol) is governed in `DATASET_GOVERNANCE.md`.

## What's in here

```
endoscopy-cade-mvp/
├── src/cade/
│   ├── data/            # public dataset loaders (Kvasir-SEG, planned: HyperKvasir)
│   ├── ingest/          # mp4 → frame pipeline + manifest schema
│   ├── models/          # YOLOv8 baseline, MedSAM-2 baseline (planned)
│   └── eval/            # sensitivity / specificity / F1 / AUC + Paris-aware
├── scripts/             # train / eval / infer CLIs
├── tests/               # pytest — metric correctness + manifest invariants
├── notebooks/           # exploratory EDA (not for production)
├── DATASET_GOVERNANCE.md # IRB pathway, anonymisation plan, proprietary asset counts
├── ROADMAP.md           # 30 / 60 / 90 day plan
└── pyproject.toml
```

## Quick start

```bash
# Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Public benchmark — Kvasir-SEG polyp segmentation
python scripts/eval_baseline.py --dataset kvasir-seg --model yolov8n-seg --weights pretrained

# Run the metric tests
pytest tests/
```

No GPU is required to run the eval against a pretrained checkpoint — it
will be slow on CPU but it will run. Training requires a single CUDA
GPU (T4 or better).

## Headline metrics we target

Against the Kvasir-SEG public test split, the bar for the baseline is:

| Metric | Target | Status |
|---|---:|---|
| Polyp Dice | ≥ 0.80 | pending first run |
| Polyp per-frame sensitivity | ≥ 0.90 | pending |
| Polyp per-frame specificity | ≥ 0.85 | pending |
| Per-clip lead time (alarm before endoscopist marks polyp) | ≥ +0.5 s | pending |

These are baseline thresholds, not state-of-the-art claims. The point of
the public eval is to make every model swap, every training tweak and
every dataset change comparable in 5 minutes by anyone reading this repo.

## Proprietary-clip eval (planned, IRB-gated)

The second eval — the one that actually matters — runs the same models
against an anonymised proprietary archive of upper-GI and colonoscopy
clips with NBI imaging, captured on Olympus CV-180 at Clínica Cirúrgica
Canova (CNPJ 45.958.018/0001-39). Counts, governance, and the IRB pathway
are in `DATASET_GOVERNANCE.md`. **No raw clips, no frames and no patient
identifiers live in this repository or in any of its history.**

## What this repo is **not**

- It is not a SaMD. It is research infrastructure. The README of any
  Streamlit / Gradio demo built on top of it carries the explicit
  "not for clinical use" banner — same convention as `dermamed` in
  the personal portfolio.
- It is not a federated-learning play. Single-site for now; federation
  is a 12-month thing if a partner site materialises.
- It is not a replacement for the endoscopist. The output of a polyp
  detector is a bounding box and a score; the diagnosis is the
  endoscopist's.

## Who's building this

Lucas Dickel Canova, MD — practising surgeon and endoscopist (CRM/RS
46.242, RQE 39.549 endoscopia), ~6,000 personal upper-GI endoscopies
and colonoscopies on Olympus CV-180 NBI since 2018, owner and technical
director of two Brazilian clinics. Solo on the build side, clinical
co-author signing the protocol on the science side.

Public portfolio: <https://www.lucascanova.com.br/portfolio>

## License

MIT. See `LICENSE`.
