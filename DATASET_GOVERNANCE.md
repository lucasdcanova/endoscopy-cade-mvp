# Dataset governance

The repository's public surface uses **only open, redistributable
datasets** (currently Kvasir-SEG; planned HyperKvasir, PolypGen). This
file describes how the **proprietary** clip archive — which never enters
the repository — is governed.

## Proprietary asset (kept out of git)

- **Source:** upper-GI endoscopies and colonoscopies performed personally
  by Lucas Dickel Canova, MD (CRM/RS 46.242, RQE 39.549) at Clínica
  Cirúrgica Canova LTDA (CNPJ 45.958.018/0001-39), Crissiumal /
  Três Passos, RS, Brazil.
- **Equipment:** Olympus CV-180 NBI tower.
- **Volume substrate:** ~6,000 upper-GI endoscopies and colonoscopies
  performed personally since 2018, of which a subset has retained video.
- **Annotation taxonomy planned:** Paris (polyp morphology), NICE
  (NBI characterisation), surface size estimate, dye-spray / biopsy /
  resection event tags.

The proprietary archive is referenced in the portfolio as
"~6,000 endoscopies — auditable under NDA" and the same NDA gate
applies to any clip-level access.

## What is NEVER in this repository

- No raw video, no extracted frames, no thumbnails
- No DICOM tags, no patient identifiers, no procedure dates
- No links pointing to a clip blob — not even an opaque URL
- No `.gitignored` `data/proprietary/` directory in any commit
- No commit message, no test fixture, no docstring referencing a
  real patient

The `git log` of this repository is checked against this rule before
every push.

## IRB and consent pathway

The proprietary eval requires an IRB protocol approved by a Brazilian
Comitê de Ética em Pesquisa (CEP) before any frame is annotated for
research use, including:

- Retroactive informed-consent waiver request OR a re-consent campaign
  for patients within the archive window
- Explicit anonymisation plan (no face, no DICOM, no procedure date,
  no operating endoscopist signature)
- Data sharing limited to single-investigator, single-site for the
  first 12 months
- All artefacts (annotations, model weights, eval reports) re-reviewed
  before public release

Status: **not yet submitted.** The repository scaffold (this commit)
predates submission on purpose — the point of the public benchmark
half is to be useful without needing the proprietary archive.

## Anonymisation pipeline (planned)

1. Strip DICOM tags + audio + any clinic-software watermark
2. Black-out fixed-position overlays (patient name strip, MRN, date)
   pre-extraction
3. Frame extraction at fixed FPS, no timestamp metadata
4. Manual second pass — clinician removes any frame with incidental
   identifiable content (face on intubation, hand tattoo, etc)
5. Annotation only against anonymised frames

The anonymisation pipeline itself will live in this repository
(`src/cade/ingest/`) so that the *method* is auditable even if the
*data* is not.

## Why this governance section exists in a research repo

Recruiters, partners, IRB reviewers and prospective collaborators all
ask the same question first: *"how did you get the data, and who could
get sued if this leaks?"*. Answering it before they ask is the
difference between a credible clinician-builder and a hobbyist with
a GitHub.

— Lucas D. Canova, MD
2026-05-28
