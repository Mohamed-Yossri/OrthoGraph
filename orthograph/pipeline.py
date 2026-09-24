from __future__ import annotations
import sqlite3
import time
from typing import TypedDict
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from .domain import Report, Tooth, Finding, assign_fdi, associate, estimate_guides, rebuild, now, review_errors


class State(TypedDict, total=False):
    case_id: str
    teeth: list
    findings: list
    runs: list
    trace: list
    options: dict
    report: dict


class Workflow:
    def __init__(self, store, vision, references):
        self.store,self.vision,self.references = store,vision,references
        self.connection = sqlite3.connect(str(store.root/'workflow.sqlite3'), check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        g = StateGraph(State)
        for name in ('detect_teeth','detect_findings','check_impaction','associate','retrieve','review','finalize'):
            g.add_node(name,getattr(self,name))
        g.add_edge(START,'detect_teeth')
        g.add_edge('detect_teeth','detect_findings')
        g.add_conditional_edges('detect_findings',self.route,{'check':'check_impaction','skip':'associate'})
        g.add_edge('check_impaction','associate')
        g.add_edge('associate','retrieve')
        g.add_edge('retrieve','review')
        g.add_edge('review','finalize')
        g.add_edge('finalize',END)
        self.graph = g.compile(checkpointer=self.checkpointer)

    def progress(self, state, stage):
        self.store.update(state['case_id'], status='processing', stage=stage)

    def event(self, state, node, detail):
        return state.get('trace',[])+[{'node':node,'detail':detail,'at':now()}]

    def detect_teeth(self, state):
        self.progress(state,'Mapping teeth')
        teeth,run = self.vision.predict('enumeration',self.store.directory(state['case_id'])/'image.png')
        return {'teeth':teeth,'runs':[run], 'trace':self.event(state,'detect_teeth',f'{len(teeth)} tooth candidates localized')}

    def detect_findings(self, state):
        self.progress(state,'Inspecting panoramic findings')
        findings,run = self.vision.predict('yolo26',self.store.directory(state['case_id'])/'image.png')
        return {'findings':findings,'runs':state['runs']+[run],
                'trace':self.event(state,'detect_findings',f'{len(findings)} findings from the panoramic detector')}

    def route(self, state):
        return 'check' if state['options'].get('sensitivity',False) or (state['options'].get('impaction',True) and any(f['label']=='third_molar' for f in state['findings'])) else 'skip'

    def check_impaction(self, state):
        self.progress(state,'Checking third-molar candidates')
        sensitivity=state['options'].get('sensitivity',False)
        findings,run = self.vision.predict('liodon',self.store.directory(state['case_id'])/'image.png',
                                          **({'include_all':True} if sensitivity else {}))
        return {'findings':state['findings']+findings, 'runs':state['runs']+[run],
                'trace':self.event(state,'check_impaction',
                    f'Liodon second detector produced {len(findings)} additional candidates; sensitivity mode={sensitivity}. All require review.')}

    def associate(self, state):
        self.progress(state,'Resolving tooth associations')
        case = self.store.get(state['case_id'])
        with Image.open(self.store.directory(state['case_id'])/'image.png') as img:
            width,height = img.size
        teeth = [Tooth(**t) for t in state['teeth']]
        findings = [Finding(**f) for f in state['findings']]
        guides = estimate_guides(teeth,width,height)
        orientation = state['options'].get('orientation','unknown')
        assign_fdi(teeth,width,height,guides,orientation)
        associate(findings,teeth)
        report = Report(case_id=state['case_id'],created_at=case['created_at'],
                        image={'width':width,'height':height,'sha256':state['options']['image_hash'],
                               'coordinate_space':'original_normalized_image_pixels','format':'PNG'},
                        orientation=orientation,guides=guides,teeth=teeth,findings=findings,
                        model_runs=state['runs'],trace=self.event(state,'associate','Assigned provisional FDI identities; ambiguous matches stay unresolved'))
        return {'report':rebuild(report).model_dump(), 'trace':report.trace}

    def retrieve(self, state):
        self.progress(state,'Linking clinical references')
        report = self.references.attach(Report(**state['report']))
        report.trace = self.event(state,'retrieve',f'{len(report.references)} applicable source records retrieved from the curated Qdrant corpus')
        report.trace.append({'node':'review','at':now(),'detail':'Workflow paused for image orientation, tooth numbering and finding review'})
        rebuild(report)
        self.store.update(state['case_id'],report=report.model_dump())
        return {'report':report.model_dump()}

    def review(self, state):
        approved = interrupt({'case_id':state['case_id'],'action':'review_evidence'})
        if approved is not True:
            raise ValueError('Review confirmation required.')
        # Read the latest persisted reviewer edits, rather than stale graph state.
        report = Report(**self.store.get(state['case_id'])['report'])
        errors = review_errors(report)
        if errors:
            raise ValueError(' '.join(errors))
        return {'report':report.model_dump()}

    def finalize(self, state):
        report = Report(**state['report'])
        report.status = 'reviewed'
        report.revision += 1
        report.trace.append({'node':'finalize','at':now(),'detail':'Reviewer completed the research review; unassigned teeth remain explicit'})
        rebuild(report)
        self.store.update(state['case_id'],status='reviewed',stage='Review complete',report=report.model_dump())
        return {'report':report.model_dump()}

    def run(self, case_id):
        try:
            case = self.store.get(case_id)
            self.graph.invoke({'case_id':case_id,'options':case['options'],'trace':[]},
                              {'configurable':{'thread_id':case_id}})
            self.store.update(case_id,status='awaiting_review',stage='Ready for review')
        except Exception as exc:
            self.store.update(case_id,status='error',stage='Processing failed',
                              error=f'{type(exc).__name__}: {str(exc)[:300]}')

    def finish(self, case_id):
        self.graph.invoke(Command(resume=True),{'configurable':{'thread_id':case_id}})
        return self.store.get(case_id)
