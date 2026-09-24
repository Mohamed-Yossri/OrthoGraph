import pytest
from pydantic import ValidationError
from domain import (Tooth, Finding, Report, ReviewPatch, assign_fdi,
    associate, rebuild, apply_patch, review_errors, estimate_guides)


def tooth(id='t1',kind=1,x=30,y=20):
    return Tooth(id=id,tooth_type=kind,bbox=[x,y,x+20,y+20],score=.8)


def report(teeth=None,findings=None,orientation='standard'):
    ts=teeth if teeth is not None else [tooth()]
    fs=findings or []
    guides={'midline':.5,'arch_y':.5}
    assign_fdi(ts,200,200,guides,orientation)
    associate(fs,ts)
    return rebuild(Report(case_id='test',created_at='2026-09-24',image={'width':200,'height':200},
                  orientation=orientation,guides=guides,teeth=ts,findings=fs,model_runs=[]))


def test_quadrants_and_midline_outward_identity():
    ts=[tooth('a',3,30,20),tooth('b',6,140,20),tooth('c',2,140,140),tooth('d',8,30,140)]
    assert [t.fdi for t in report(ts).teeth]==[13,26,32,48]


def test_missing_incisor_does_not_shift_other_teeth():
    assert report([tooth(kind=3)]).teeth[0].fdi==13


def test_unknown_orientation_abstains():
    r=report(orientation='unknown')
    assert r.teeth[0].fdi is None
    assert all(x['status']=='not_detected' for x in r.odontogram.values())


def test_flipped_orientation_swaps_sides():
    assert report(orientation='flipped').teeth[0].fdi==21


def test_duplicate_fdi_is_preserved_and_blocks_confirmation():
    r=report([tooth('a'),tooth('b',x=60)])
    assert all('duplicate_fdi' in t.flags for t in r.teeth)
    with pytest.raises(ValueError,match='duplicate'):
        apply_patch(r,ReviewPatch(expected_revision=0,numbering_confirmed=True))


def test_unassigning_duplicate_resolves_conflict():
    r=report([tooth('a'),tooth('b',x=60)])
    r=apply_patch(r,ReviewPatch(expected_revision=0,tooth_assignments={'b':None}))
    assert all('duplicate_fdi' not in t.flags for t in r.teeth)
    assert r.teeth[1].fdi is None


def test_invalid_fdi_rejected():
    with pytest.raises(ValidationError):
        ReviewPatch(expected_revision=0,tooth_assignments={'a':19})


def test_stale_review_cannot_overwrite():
    with pytest.raises(ValueError,match='another window'):
        apply_patch(report(),ReviewPatch(expected_revision=3))


def test_unknown_tooth_id_rejected():
    with pytest.raises(ValueError,match='Unknown tooth'):
        apply_patch(report(),ReviewPatch(expected_revision=0,tooth_assignments={'bad':11}))


def test_bridge_remains_region_instead_of_forced_tooth():
    f=Finding(id='f',label='bridge',title='Bridge',category='restoration',raw_label='Ceramic Bridge',
              model='yolo26',bbox=[30,20,50,40],score=.9)
    r=report(findings=[f]);assert r.findings[0].tooth_id is None
    assert r.findings[0].association=='region'


def test_fdi_correction_updates_findings_and_invalidates_decision():
    f=Finding(id='f',label='filling',title='Filling',category='restoration',raw_label='Dental Filling',
              model='yolo26',bbox=[32,23,45,35],score=.9,review='accepted')
    r=report(findings=[f]);assert r.findings[0].fdi==11
    r=apply_patch(r,ReviewPatch(expected_revision=0,tooth_assignments={'t1':12}))
    assert r.findings[0].fdi==12 and r.findings[0].review=='unreviewed'
    assert r.odontogram['11']['status']=='not_detected'


def test_rejected_finding_stays_in_audit_not_odontogram():
    f=Finding(id='f',label='filling',title='Filling',category='restoration',raw_label='Dental Filling',
              model='yolo26',bbox=[32,23,45,35],score=.9)
    r=apply_patch(report(findings=[f]),ReviewPatch(expected_revision=0,finding_reviews={'f':'rejected'}))
    assert r.findings[0].review=='rejected'
    assert not r.odontogram['11']['finding_ids']
    assert r.trace[-1]['changes']['finding_reviews']=={'f':'rejected'}


def test_review_requires_orientation_numbering_and_decisions():
    assert len(review_errors(report()))==2


def test_manual_numbering_survives_non_geometry_edit():
    r=apply_patch(report(),ReviewPatch(expected_revision=0,tooth_assignments={'t1':14}))
    r=apply_patch(r,ReviewPatch(expected_revision=1,orientation_confirmed=True))
    assert r.teeth[0].fdi==14


def test_midline_uses_incisors_when_dentition_asymmetric():
    ts=[tooth('a',1,85,20),tooth('b',1,105,20),tooth('c',8,5,20),tooth('d',1,85,150),tooth('e',1,105,150)]
    assert estimate_guides(ts,200,200)['midline']==pytest.approx(.525)


def test_third_molar_not_attached_to_nearby_second_molar():
    f=Finding(id='f',label='third_molar',title='Third molar',category='anatomy',raw_label='Wisdom Tooth',
              model='yolo26',bbox=[30,20,50,40],score=.9)
    r=report([tooth(kind=7)],findings=[f])
    assert r.findings[0].tooth_id is None


def test_impaction_not_attached_to_partial_neighbor_overlap():
    f=Finding(id='f',label='impaction_candidate',title='Impaction candidate',category='pathology',
              raw_label='impacted_tooth',model='liodon',bbox=[25,15,35,25],score=.8)
    r=report([tooth(kind=7)],findings=[f]);assert r.findings[0].tooth_id is None
