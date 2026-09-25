# OrthoGraph continuation memory

Updated: 2026-09-24. Read this with `README.md` and the running code. Keep this file current after material changes so another AI can continue the project.

## Intent and scope

Build a credible, impressive OPG research workbench that makes model evidence inspectable and human corrections durable. The current application is a research prototype, not a clinically validated diagnostic system. The GitHub repository is the canonical source and uses a flat root layout (`app.py`, `pipeline.py`, `static/`, `tests/`, `research/`, `docs/`); the live Lightning Studio currently serves a separate working copy at `/teamspace/studios/this_studio/orthograph`. Sync tested source edits to that copy when updating the live app. Never copy runtime case data, credentials, model weights, or the upstream example image into Git.

## Current architecture

FastAPI serves custom responsive HTML/CSS/JS. The case graph is `detect_teeth → detect_findings → optional check_impaction → associate → retrieve → review interrupt → finalize`. Tooth detection uses OPGAgent eight-class enumeration, finding detection uses panoramic YOLO26, and Liodon is a conditional second detector. `domain.py` proposes FDI numbers using predicted tooth type plus quadrant and orientation; uncertainty, duplicate labels, and unresolved regional associations are explicit. Qdrant local retrieval contains four curated source-linked summaries. LangGraph checkpoints and cases persist in SQLite; images remain on local disk. The report and UI retain model runs, coordinates, reviewer edits, evidence links, and JSON export. Optional Gemini crop inspection is user-triggered and cannot change review decisions; live successful generation remains unverified.

The service has Argon2 password hashes, revocable sessions, account-isolated cases, a bounded GPU queue, and user registration without email. It is a single-instance demo, not a clinical multi-tenant deployment. Use only authorized, de-identified adult permanent-dentition panoramic images. The upstream research example is downloaded separately and is not redistributed in Git.

## Verification and gaps

The Python API/domain tests passed 28 checks on Python 3.12. GitHub Actions runs CPU-only tests. A 50-image DENTEX exploratory audit reported 180/182 annotated abnormal tooth boxes localized at IoU ≥ 0.3 and exact FDI for 158/180 localized boxes; YOLO26 caries site hits were 24/133 and Liodon 46/133 with more unmatched boxes. These are site-matching observations, not clinical performance metrics or an independent test. Full definitions are in `docs/WORKFLOW.md`; machine-readable results are in `research/dentex_validation.json`.

There is no validated lesion segmentation, bone-loss measurement, pulp assessment, periodontal staging, DICOM/PHI pipeline, CBCT support, model training, external clinical validation, or production deployment. Next technical priority: build a source-disjoint expert-labelled evaluation with per-class false positives, tooth/FDI/association metrics, abstention coverage, and reviewer correction burden. Add modules only when their evaluation supports the added complexity.

## Operations

From repository root: `python -m pip install -r requirements.txt`, `python download_assets.py`, `python main.py`. Tests: `python -m pytest tests -q`. The live app uses Lightning port 7860 and can auto-start after Studio sleep; allow for cold start and available GPU credits. Public project entry point: https://mohamed-yossri.github.io/OrthoGraph/ . Its GitHub Pages page is only a static link to the Lightning backend.

Update this memory after architectural or deployment changes, keeping it safe for a public repository. Do not write secrets, private case details, or security incident details here.

## Repository layout update

The earlier nested `orthograph/` source wrapper was removed at the user's request. Python modules and requirements now live at repository root; tests and static/research assets are root directories. `main.py` imports `app:create_app`. GitHub Actions and all documented run commands use root paths. The live Lightning deployment still runs its separate working copy; this source-only layout change does not alter that deployment. Keep both copies aligned when making behavioral changes.

## High-bit-depth upload fix (2026-09-25)

A user-submitted panoramic PNG produced an almost-white saved case image and no detections. The saved image was 99% near-white; its original upload bytes were not retained, so its bit depth cannot be confirmed. Reproduced a likely root cause: direct Pillow `I;16` to RGB conversion clips nearly all 16-bit intensities to white. Upload now percentile-normalizes 16-bit grayscale PNGs to 8-bit before RGB conversion and records source mode/normalization in report image metadata. A separate quality gate rejects images with almost no midtone detail before inference. Added two API tests; suite passed 28 tests. The fix was copied to the separate live source, its process restarted, and public `/login` returned 200. Existing damaged case images cannot be reconstructed from saved 8-bit PNGs; re-upload the original source file.

## Analysis cap removed (2026-09-25)

At the user's request, the three-analyses-per-account-per-day limit was removed. `app.py` no longer calls a usage reservation, `storage.py` no longer creates or writes to the usage table, and the regression test now confirms a registered account can submit four analyses in one day. The existing `analysis_usage` table in old SQLite databases is inert and is not read. The bounded GPU queue remains for concurrency control. Canonical and live-copy test suites passed 28 tests each; live app was restarted and `/login` returned 200.
