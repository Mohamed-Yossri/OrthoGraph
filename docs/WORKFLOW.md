# Detailed workflow and validation

OrthoGraph is an interactive research workbench for dental panoramic radiographs. Local vision models produce tooth and finding candidates; an editable odontogram links every observation to its image region, model provenance and applicable reference context.

## Run from the repository root

The application is served on **port 7860**. The live Lightning URL is linked from the repository README. For the first sign-in, read the one-time credential in `data/initial-login.txt` and set your own password on the sign-in page. That file is deleted after setup. Use the **Explore the research example** button to try the full workflow.

```bash
# From the repository root
python main.py
```

The current Lightning Studio has CUDA PyTorch and the required checkpoints in its live working copy. For a fresh checkout:

```bash
python -m pip install -r requirements.txt
python download_assets.py
# Optional upstream example for local research testing:
python download_assets.py --demo
python main.py
```

Python 3.12, PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128 were used on NVIDIA L4. Install an appropriate PyTorch/torchvision build separately on a different host. A CPU fallback exists but was not latency-benchmarked. Lightning permits one conda environment, so dependencies are installed in the Studio environment. FastEmbed downloads a small BGE embedding model on first reference retrieval and then caches it locally.

## What works

- PNG/JPEG upload, input limits, contrast validation and metadata-stripped image normalization. An optional sensitivity mode adds Liodon caries/periapical leads, with an explicit increase in expected false alarms.
- Full-OPG tooth-type detection (OPGAgent), nine-class findings (YOLO26), and conditional impaction checking (Liodon).
- Interactive image zoom/pan, selectable regions, contextual crops and 32-slot FDI odontogram.
- Provisional quadrant assignment using geometry; explicit unknown orientation, duplicate numbering, unassigned teeth and region-level findings.
- Individual tooth correction, finding-to-tooth reassignment, accept/reject decisions and notes.
- Persistent LangGraph interrupts/resume using SQLite, including server restarts during human review.
- Qdrant local semantic retrieval using BGE-small embeddings and a curated starter corpus.
- Optional, explicitly requested Gemini crop descriptions with structured output validation; these cannot overwrite review decisions.
- Source-linked report, typed evidence graph, workflow trace and JSON download.
- Case history and deletion of images, reports and workflow checkpoints.
- Responsive desktop/mobile UI; username/password registration and sign-in with Argon2 password hashing, revocable sessions, and case isolation.

## Review workflow

1. Upload a de-identified, adult permanent-dentition OPG or open the research example.
2. Expand **Review setup**, select image orientation and adjust the provisional midline/arch separator if necessary. Imported images start with unknown orientation. No automatic left/right flip is applied.
3. Inspect the odontogram and assignment-queue chips. Select a tooth to correct its FDI number or leave it unassigned. Duplicate assignments must be resolved before confirming numbering. Geometry changes reset manual assignments; affected finding decisions must be reviewed again.
4. Select each finding. Inspect its crop, adjust its associated tooth, add a note, then accept or reject the candidate. Acceptance is a reviewer decision within a research tool, not a certified diagnosis.
5. Confirm orientation and numbering, then **Complete review**. Unassigned teeth can remain unresolved. Download JSON at any point; draft/reviewed status is explicit.

The model's 1–8 tooth classes are identities, not detection ranks. A missing incisor does not shift all other numbers. A model non-detection is not relabelled as healthy or missing. Wisdom teeth and impaction remain separate findings. Bridges, implant candidates and missing-tooth candidates default to region-level observations rather than being forced onto a visible neighboring tooth.

## Configuration

| Variable | Purpose |
|---|---|
| `PORT` | HTTP port; default `7860` |
| `ORTHOGRAPH_HOST` | Bind address; default `0.0.0.0` |
| `ORTHOGRAPH_DATA_DIR` | Case/checkpoint/cache directory; default `data` |
| `GEMINI_API_KEY` | Enables optional crop inspection |
| `ORTHOGRAPH_GEMINI_MODEL` | Override the Gemini model for your account |
| `HF_TOKEN` | Optional for Hugging Face access; detector files are public |

The owner username is `mohamed-yossri`. The first server launch writes `data/initial-login.txt` (mode 0600); use its temporary password on the sign-in page, which then prompts for a new password. Other users can create separate accounts at `/register` without an email address or verification. Passwords are stored only as Argon2 hashes in `data/auth.sqlite3`. Sessions are server-side and revocable. Existing cases migrate to the owner account; newly registered users cannot list or open them. Registration is limited to 25 accounts and five attempts per IP per hour; demo accounts can run three analyses per day. Email-based password recovery is not implemented. Use Lightning HTTPS or a TLS reverse proxy for remote access. This is a single-user research service, not a multi-tenant clinical deployment. Do not run multiple Uvicorn workers against Qdrant local mode. One GPU worker and a four-case queue bound inference concurrency.

Each Gemini action explicitly asks to send the selected crop to Google. Maximum three requests per case, one concurrent request, 45-second timeout, and a validated JSON schema. No automatic cloud upload occurs. API quota/provider failures leave local review available. In this session, model listing succeeded, but live synthetic-image generation returned HTTP 503 on the tested current models; the integration is implemented and schema/error handling is tested, while successful live crop generation remains unverified. The default is `gemini-3.6-flash`, as recommended by the provider after it rejected `gemini-2.5-flash` for this account. API keys are read from environment secrets, not embedded in frontend code or URLs.

## References and provenance

The initial corpus contains **four curated summaries**, not full ingested guideline PDFs:

- ADA caries assessment and classification context.
- AAE expert discussion of periapical differential diagnosis (labelled an expert article, not a formal guideline).
- EFP periodontal classification context.
- NICE TA1 third-molar guidance, explicitly labelled with its NHS context.

Each summary has a source URL, publisher, section, review date and content hash. Retrieval is filtered for applicable finding types. Unsupported findings produce no invented reference or treatment recommendation. Corpus coverage and clinical applicability need expansion and expert review. The reference library allows semantic search of the same index.

`research/assets.json` records immutable checkpoint URLs and SHA-256 hashes. Checkpoint hashes are verified before loading. The JSON report retains source model labels, coordinates, scores, transforms, associations, review decisions and graph edges. Source links contextualize a finding; they do not prove it is correct.

## Tests and reproduction

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python verify_assets.py
```

Tests cover orientation, missing teeth, duplicate identities, uncertain associations, changed-decision invalidation, input validation, stale edits, checkpoint recovery across restart, finalization, export, deletion, account migration, sign-up and case isolation, and VLM failure handling. GPU smoke tests and browser tests are additional integration checks; they are not clinical accuracy studies.

Runtime records: `research/verification/report.json`. Browser captures: `research/screenshots/`. Server log and PID: `data/server.log`, `data/server.pid`.

## Exploratory DENTEX model audit

Run `python evaluate_dentex.py`. The validation archive is downloaded at a pinned revision, SHA-256 verified, and kept under ignored `data/dentex/`. The 50-image, 182-abnormal-tooth result is in `research/dentex_validation.json` and shown in the app's **Model audit** view.

| Metric | Result |
|---|---:|
| Abnormal tooth boxes localized at IoU ≥ 0.3 | 180/182 |
| Exact FDI among localized abnormal teeth | 158/180 |
| YOLO26 caries/deep-caries site hits | 24/133, with 4 unmatched boxes |
| Liodon caries/deep-caries site hits | 46/133, with 37 unmatched boxes |
| YOLO26 periapical site hits | 4/9, with 1 unmatched box |
| Liodon periapical site hits | 2/9, with 0 unmatched boxes |
| Liodon impaction site hits | 33/40, with 11 unmatched boxes |

**Metric definition:** A lesion detection counts as a hit if its box center falls inside an annotated DENTEX abnormal-tooth box, matched once per category and site. This differs from mAP because the published models may draw small lesion boxes while DENTEX labels entire teeth. Caries and deep caries are merged. The tooth labels cover abnormal teeth only, so these data cannot establish whole-mouth tooth recall. The split is small, and patient/source overlap with model training was not audited. Liodon's checkpoint was already selected using DENTEX validation, making those numbers especially unsuitable as independent performance estimates. These results motivated an optional sensitivity mode; they do not supply a clinical operating threshold.

## Model attribution and limitations

- [OPGAgent](https://github.com/Zhaolin-Yu/OPGAgent): eight-class tooth enumeration checkpoint. Root code license and enumeration README differ; review upstream model/data terms before redistributing weights.
- [EuricoGVP/YOLO26_Dental_Detection](https://huggingface.co/EuricoGVP/YOLO26_Dental_Detection): primary findings model; labelled AGPL-3.0 and research-only intended use.
- [Liodon panoramic detector](https://huggingface.co/liodon-ai/dental-panoramic-detector): impaction tool; labelled CC-BY-NC-4.0.
- [BAAI BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): reference embeddings.

Weights and the upstream example image are excluded from version control. A public repository's sample image is not proof of unrestricted redistribution rights. Model and data licenses do not automatically inherit any future application-code license.

This release is a working research prototype, not a clinically validated system. It does not diagnose pulp status, caries depth or periodontal stage. It does not support DICOM, mixed dentition or automatic bone-loss measurement. Tooth geometry can fail with tilted heads, atypical anatomy and missing teeth; the correction workflow is intentional. There is no automatic recognition that an uploaded image is an OPG, so that modality is an input requirement. Fine-grained pathology accuracy and FDI accuracy still require a labelled, source-disjoint evaluation.

A frozen external evaluation, per-class sensitivity/false positives, association accuracy, abstention coverage and reviewer correction burden are the next priorities. See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the original design rationale. The repository-root [`memory.md`](../memory.md) is the continuation handoff for another AI.
