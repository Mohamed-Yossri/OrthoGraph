from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

FDI = tuple(q * 10 + t for q in range(1, 5) for t in range(1, 9))
LABELS = {
    'Apical Periodontitis': ('periapical_radiolucency', 'Periapical lesion candidate', 'pathology'),
    'Decay': ('suspected_caries', 'Caries candidate', 'pathology'),
    'Wisdom Tooth': ('third_molar', 'Third molar', 'anatomy'),
    'Missing Tooth': ('missing_tooth_candidate', 'Missing tooth candidate', 'anatomy'),
    'Dental Filling': ('filling', 'Dental filling', 'restoration'),
    'Root Canal Filling': ('root_canal_filling', 'Root canal filling', 'restoration'),
    'Implant': ('implant', 'Implant', 'restoration'),
    'Porcelain Crown': ('crown', 'Crown', 'restoration'),
    'Ceramic Bridge': ('bridge', 'Bridge', 'restoration'),
    'impacted_tooth': ('impaction_candidate', 'Impaction candidate', 'pathology'),
    'caries': ('suspected_caries', 'Caries candidate', 'pathology'),
    'periapical_lesion': ('periapical_radiolucency', 'Periapical lesion candidate', 'pathology'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Tooth(StrictModel):
    id: str
    tooth_type: int = Field(ge=1, le=8)
    fdi: int | None = None
    candidate_fdi: int | None = None
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float = Field(ge=0, le=1)
    flags: list[str] = Field(default_factory=list)
    assignment_source: str = 'geometry_and_type'

    @field_validator('fdi', 'candidate_fdi')
    @classmethod
    def valid_fdi(cls, v):
        if v is not None and v not in FDI:
            raise ValueError('FDI must be a permanent tooth number 11–48 (digits 1–8).')
        return v


class Finding(StrictModel):
    id: str
    label: str
    title: str
    category: str
    raw_label: str
    model: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float = Field(ge=0, le=1)
    tooth_id: str | None = None
    fdi: int | None = None
    candidate_teeth: list[str] = Field(default_factory=list)
    association: str = 'unresolved'
    review: Literal['unreviewed', 'accepted', 'rejected'] = 'unreviewed'
    notes: str = ''
    reference_ids: list[str] = Field(default_factory=list)
    vlm: dict | None = None


class Report(StrictModel):
    schema_version: str = '1.0'
    case_id: str
    created_at: str
    status: Literal['awaiting_review', 'reviewed'] = 'awaiting_review'
    image: dict
    orientation: Literal['standard', 'flipped', 'unknown'] = 'unknown'
    orientation_confirmed: bool = False
    numbering_confirmed: bool = False
    guides: dict
    teeth: list[Tooth]
    findings: list[Finding]
    odontogram: dict = Field(default_factory=dict)
    references: list[dict] = Field(default_factory=list)
    considerations: list[dict] = Field(default_factory=list)
    model_runs: list[dict]
    trace: list[dict] = Field(default_factory=list)
    evidence_graph: dict = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=lambda: [
        'Research output; not a clinical diagnosis. All model findings require review.',
        'Non-detection does not establish a healthy or missing tooth.',
        'Adult permanent dentition only. Tooth numbers are provisional until reviewed.',
        'Pulp status, caries depth and periodontal stage are not assessed.',
        'Detector scores are not calibrated disease probabilities.',
    ])
    revision: int = 0


class ReviewPatch(StrictModel):
    expected_revision: int = Field(ge=0)
    orientation: Literal['standard', 'flipped', 'unknown'] | None = None
    orientation_confirmed: bool | None = None
    numbering_confirmed: bool | None = None
    midline: float | None = Field(default=None, ge=.1, le=.9)
    arch_y: float | None = Field(default=None, ge=.1, le=.9)
    tooth_assignments: dict[str, int | None] = Field(default_factory=dict)
    finding_reviews: dict[str, Literal['unreviewed', 'accepted', 'rejected']] = Field(default_factory=dict)
    finding_teeth: dict[str, str | None] = Field(default_factory=dict)
    notes: dict[str, str] = Field(default_factory=dict)

    @field_validator('tooth_assignments')
    @classmethod
    def valid_numbers(cls, values):
        if any(v is not None and v not in FDI for v in values.values()):
            raise ValueError('Invalid permanent FDI number.')
        return values

    @field_validator('notes')
    @classmethod
    def note_lengths(cls, values):
        if any(len(v) > 2000 for v in values.values()):
            raise ValueError('Notes must be under 2,000 characters.')
        return values


def center(box):
    return (box[0]+box[2])/2, (box[1]+box[3])/2


def estimate_guides(teeth, width, height):
    if not teeth:
        return {'midline': .5, 'arch_y': .5, 'method': 'image_center_fallback'}
    xs = [center(t.bbox)[0] for t in teeth]
    ys = [center(t.bbox)[1] for t in teeth]
    # Jaw separation is a provisional estimate; it is explicitly editable in the UI.
    lo, hi = min(ys), max(ys)
    for _ in range(12):
        upper = [y for y in ys if abs(y-lo) <= abs(y-hi)]
        lower = [y for y in ys if abs(y-lo) > abs(y-hi)]
        if not upper or not lower:
            break
        lo, hi = sum(upper)/len(upper), sum(lower)/len(lower)
    midpoints = []
    for jaw in (True, False):
        incisors = [center(t.bbox)[0] for t in teeth if t.tooth_type == 1
                    and (center(t.bbox)[1] < (lo+hi)/2) == jaw]
        if len(incisors) == 2:
            midpoints.append(sum(incisors)/2)
    midline = statistics.median(midpoints) if midpoints else (min(xs)+max(xs))/2
    return {'midline': max(.1, min(.9, midline/width)),
            'arch_y': max(.1, min(.9, (lo+hi)/2/height)), 'method': 'provisional_two_arch_estimate'}


def assign_fdi(teeth, width, height, guides, orientation):
    mid = guides['midline'] * width
    arch = guides['arch_y'] * height
    for tooth in teeth:
        tooth.flags = []
        if tooth.assignment_source == 'reviewer':
            continue
        x, y = center(tooth.bbox)
        left, upper = x < mid, y < arch
        if orientation == 'flipped':
            left = not left
        quadrant = (1 if left else 2) if upper else (4 if left else 3)
        tooth.candidate_fdi = quadrant * 10 + tooth.tooth_type
        tooth.fdi = tooth.candidate_fdi if orientation != 'unknown' else None
        if orientation == 'unknown':
            tooth.flags.append('orientation_unconfirmed')
        if abs(y-arch) < height * .035:
            tooth.flags.append('arch_boundary')
            tooth.fdi = None
    groups = {}
    for t in teeth:
        if t.fdi is not None:
            groups.setdefault(t.fdi, []).append(t)
    for group in groups.values():
        if len(group) > 1:
            for t in group:
                t.flags.append('duplicate_fdi')
    return teeth


def association_candidates(finding, teeth):
    fx, fy = center(finding.bbox)
    x1,y1,x2,y2 = finding.bbox
    area = max(1, (x2-x1)*(y2-y1))
    candidates = []
    for t in teeth:
        if finding.label == 'third_molar' and t.tooth_type != 8:
            continue
        tx,ty = center(t.bbox)
        a,b,c,d = t.bbox
        overlap = max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/area
        if finding.label in ('third_molar', 'impaction_candidate'):
            tooth_coverage = overlap * area / max(1, (c-a)*(d-b))
            if tooth_coverage < .5:
                continue
        distance = math.hypot((fx-tx)/max(1,c-a),(fy-ty)/max(1,d-b))
        if overlap > .1 or distance < 1.1:
            candidates.append((overlap + max(0,1-distance)*.3, t))
    return sorted(candidates, key=lambda pair: pair[0], reverse=True)


def associate(findings, teeth):
    index = {t.id: t for t in teeth}
    for f in findings:
        if f.association == 'reviewer':
            f.fdi = index[f.tooth_id].fdi if f.tooth_id else None
            continue
        f.tooth_id, f.fdi = None, None
        ranked = association_candidates(f, teeth)
        f.candidate_teeth = [t.id for _, t in ranked[:3]]
        # These are region/site findings, not reliably attributable to one visible tooth.
        if f.label in ('bridge', 'missing_tooth_candidate', 'implant'):
            f.association = 'region'
            continue
        if ranked and ranked[0][0] > .4 and (len(ranked)==1 or ranked[0][0]-ranked[1][0] > .15):
            tooth = ranked[0][1]
            f.tooth_id, f.fdi = tooth.id, tooth.fdi
            f.association = 'provisional'
        else:
            f.association = 'unresolved'
    return findings


def rebuild(report):
    report.odontogram = {str(n): {'status': 'not_detected', 'tooth_ids': [], 'finding_ids': []} for n in FDI}
    for t in report.teeth:
        if t.fdi is not None:
            slot = report.odontogram[str(t.fdi)]
            slot['tooth_ids'].append(t.id)
            slot['status'] = 'conflict' if len(slot['tooth_ids']) > 1 else 'detected'
    for f in report.findings:
        if f.fdi is not None and f.review != 'rejected':
            report.odontogram[str(f.fdi)]['finding_ids'].append(f.id)
    nodes = [{'id': 'image', 'type': 'image'}]
    edges = []
    for run in report.model_runs:
        nodes.append({'id': run['id'], 'type': 'model_run', 'model': run['model']})
    for t in report.teeth:
        nodes.append({'id': t.id, 'type': 'tooth', 'fdi': t.fdi})
        edges.extend([{'source':t.id,'target':'image','type':'observed_in'},
                      {'source':t.id,'target':'enumeration','type':'produced_by'}])
    for f in report.findings:
        nodes.append({'id': f.id, 'type': 'finding', 'label': f.label, 'review': f.review})
        edges.extend([{'source':f.id,'target':'image','type':'observed_in'},
                      {'source':f.id,'target':f.model,'type':'produced_by'}])
        if f.tooth_id:
            edges.append({'source': f.id, 'target': f.tooth_id, 'type': f.association+'_association'})
        for ref in f.reference_ids:
            edges.append({'source': f.id, 'target': ref, 'type': 'contextualized_by'})
    nodes.extend({'id':r['id'],'type':'reference','title':r['title']} for r in report.references)
    report.evidence_graph = {'nodes': nodes, 'edges': edges}
    return report


def apply_patch(report: Report, patch: ReviewPatch):
    if report.status == 'reviewed':
        raise ValueError('This review is finalized. Start a new case to re-analyze.')
    if patch.expected_revision != report.revision:
        raise ValueError('This case changed in another window. Reload before saving.')
    result = report.model_copy(deep=True)
    original_associations = {f.id: (f.tooth_id, f.fdi) for f in result.findings}
    teeth = {t.id:t for t in result.teeth}
    findings = {f.id:f for f in result.findings}
    if set(patch.tooth_assignments)-teeth.keys():
        raise ValueError('Unknown tooth identifier.')
    if (set(patch.finding_reviews)|set(patch.finding_teeth)|set(patch.notes))-findings.keys():
        raise ValueError('Unknown finding identifier.')
    if any(t is not None and t not in teeth for t in patch.finding_teeth.values()):
        raise ValueError('Unknown associated tooth.')
    changed_geometry = patch.orientation is not None or patch.midline is not None or patch.arch_y is not None
    if patch.orientation is not None:
        result.orientation = patch.orientation
        result.orientation_confirmed = False
    if changed_geometry:
        result.numbering_confirmed = False
        for t in result.teeth:
            t.assignment_source = 'geometry_and_type'
        for f in result.findings:
            if f.association == 'reviewer':
                f.association = 'unresolved'
    for key in ('midline', 'arch_y'):
        value = getattr(patch,key)
        if value is not None:
            result.guides[key] = value
    for tid, fdi in patch.tooth_assignments.items():
        teeth[tid].fdi = fdi
        teeth[tid].assignment_source = 'reviewer'
        result.numbering_confirmed = False
    assign_fdi(result.teeth, result.image['width'], result.image['height'], result.guides, result.orientation)
    for fid, tid in patch.finding_teeth.items():
        findings[fid].tooth_id = tid
        findings[fid].association = 'reviewer'
    associate(result.findings, result.teeth)
    for f in result.findings:
        if (f.tooth_id, f.fdi) != original_associations[f.id]:
            f.review = 'unreviewed'
    for fid, value in patch.finding_reviews.items():
        findings[fid].review = value
    for fid, value in patch.notes.items():
        findings[fid].notes = value
    if patch.orientation_confirmed is not None:
        if patch.orientation_confirmed and result.orientation == 'unknown':
            raise ValueError('Select a known orientation before confirming it.')
        result.orientation_confirmed = patch.orientation_confirmed
    if patch.numbering_confirmed is not None:
        if patch.numbering_confirmed and any('duplicate_fdi' in t.flags for t in result.teeth):
            raise ValueError('Resolve duplicate tooth numbers before confirming numbering.')
        result.numbering_confirmed = patch.numbering_confirmed
    result.revision += 1
    result.trace.append({'node':'review_edit','at':now(),'detail':'Saved reviewer edits',
                         'changes':patch.model_dump(exclude_none=True)})
    return rebuild(result)


def review_errors(report):
    errors = []
    if not report.orientation_confirmed:
        errors.append('Confirm image orientation.')
    if not report.numbering_confirmed:
        errors.append('Review and confirm tooth numbering; unresolved teeth may remain unassigned.')
    if any('duplicate_fdi' in t.flags for t in report.teeth):
        errors.append('Resolve duplicate FDI assignments.')
    if any(f.review=='unreviewed' for f in report.findings):
        errors.append('Accept or reject every finding candidate.')
    return errors
