# OrthoGraph continuation memory

Updated: 2026-09-24. Read this with `README.md` and the running code. Keep this file current after material changes so another AI can continue the project.

## Intent and scope

Build a credible, impressive OPG research workbench that makes model evidence inspectable and human corrections durable. The current application is a research prototype, not a clinically validated diagnostic system. The GitHub repository is the canonical source; the live Lightning Studio currently serves a separate working copy at `/teamspace/studios/this_studio/orthograph`. Sync tested source edits to that copy when updating the live app. Never copy runtime case data, credentials, model weights, or the upstream example image into Git.

## Current architecture

FastAPI serves custom responsive HTML/CSS/JS. The case graph is `detect_teeth → detect_findings → optional check_impaction → associate → retrieve → review interrupt → finalize`. Tooth detection uses OPGAgent eight-class enumeration, finding detection uses panoramic YOLO26, and Liodon is a conditional second detector. `domain.py` proposes FDI numbers using predicted tooth type plus quadrant and orientation; uncertainty, duplicate labels, and unresolved regional associations are explicit. Qdrant local retrieval contains four curated source-linked summaries. LangGraph checkpoints and cases persist in SQLite; images remain on local disk. The report and UI retain model runs, coordinates, reviewer edits, evidence links, and JSON export. Optional Gemini crop inspection is user-triggered and cannot change review decisions; live successful generation remains unverified.

The service has Argon2 password hashes, revocable sessions, account-isolated cases, a bounded GPU queue, and user registration without email. It is a single-instance demo, not a clinical multi-tenant deployment. Use only authorized, de-identified adult permanent-dentition panoramic images. The upstream research example is downloaded separately and is not redistributed in Git.

## Verification and gaps

The Python API/domain tests passed 26 checks on Python 3.12. GitHub Actions runs CPU-only tests. A 50-image DENTEX exploratory audit reported 180/182 annotated abnormal tooth boxes localized at IoU ≥ 0.3 and exact FDI for 158/180 localized boxes; YOLO26 caries site hits were 24/133 and Liodon 46/133 with more unmatched boxes. These are site-matching observations, not clinical performance metrics or an independent test. Full definitions are in `orthograph/README.md`; machine-readable results are in `orthograph/research/dentex_validation.json`.

There is no validated lesion segmentation, bone-loss measurement, pulp assessment, periodontal staging, DICOM/PHI pipeline, CBCT support, model training, external clinical validation, or production deployment. Next technical priority: build a source-disjoint expert-labelled evaluation with per-class false positives, tooth/FDI/association metrics, abstention coverage, and reviewer correction burden. Add modules only when their evaluation supports the added complexity.

## Operations

From repository root: `python -m pip install -r orthograph/requirements.txt`, `python -m orthograph.download_assets`, `python main.py`. Tests: `python -m pytest orthograph/tests -q`. The live app uses Lightning port 7860 and can auto-start after Studio sleep; allow for cold start and available GPU credits. Public project entry point: https://mohamed-yossri.github.io/OrthoGraph/ . Its GitHub Pages page is only a static link to the Lightning backend.

Update this memory after architectural or deployment changes, keeping it safe for a public repository. Do not write secrets, private case details, or security incident details here.
