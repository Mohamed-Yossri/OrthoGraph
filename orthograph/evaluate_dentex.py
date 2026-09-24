"""Exploratory DENTEX validation audit; never interpret as clinical accuracy."""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import requests
from .vision import Vision
from .domain import Tooth, Finding, assign_fdi, estimate_guides, center

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/dentex'
REV='7b27ccc8e342dcb774f69adc6ca5e6c09fefce93'
BASE=f'https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/{REV}/DENTEX/'
ZIP_SHA='6370bb4f1024bd610cde13242a465cb2eff195fc02f56ac22126555e7edc7bc3'
ANNOTATION_SHA='d058afd35d2849923c7c045e61fd3e05d231dcf74d55009993fff88bd9b6f5a2'
CATEGORY={0:'impaction_candidate',1:'suspected_caries',2:'periapical_radiolucency',3:'suspected_caries'}
CATEGORIES=('impaction_candidate','suspected_caries','periapical_radiolucency')


def download():
    DATA.mkdir(parents=True,exist_ok=True)
    zippath=DATA/'validation_data.zip'
    if not zippath.exists():
        with requests.get(BASE+'validation_data.zip',stream=True,timeout=120) as response:
            response.raise_for_status()
            with zippath.open('wb') as file:
                for chunk in response.iter_content(1<<20):file.write(chunk)
    if hashlib.sha256(zippath.read_bytes()).hexdigest()!=ZIP_SHA:
        raise RuntimeError('DENTEX validation archive hash mismatch')
    annotation=DATA/'validation_triple.json'
    if not annotation.exists():
        r=requests.get(BASE+'validation_triple.json',timeout=30);r.raise_for_status()
        annotation.write_bytes(r.content)
    if hashlib.sha256(annotation.read_bytes()).hexdigest()!=ANNOTATION_SHA:
        raise RuntimeError('DENTEX validation annotations hash mismatch')
    return zippath,json.loads(annotation.read_text())


def images(zippath,descriptions):
    root=DATA/'images';root.mkdir(exist_ok=True)
    with ZipFile(zippath) as archive:
        names={info.filename:info for info in archive.infolist()}
        for item in descriptions:
            name=item['file_name']
            if '/' in name or '..' in name:
                raise RuntimeError('Unexpected DENTEX image name')
            dest=root/name
            if not dest.exists():
                source='validation_data/quadrant_enumeration_disease/xrays/'+name
                info=names.get(source)
                if not info or info.file_size>25*1024*1024:
                    raise RuntimeError('Invalid archive member '+source)
                with archive.open(info) as image:dest.write_bytes(image.read())
            yield item,dest


def intersection(box1,box2):
    x1,y1,x2,y2=box1;a,b,c,d=box2
    return max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))


def iou(box1,box2):
    inter=intersection(box1,box2)
    if not inter:return 0.
    a=max(1,(box1[2]-box1[0])*(box1[3]-box1[1]))
    b=max(1,(box2[2]-box2[0])*(box2[3]-box2[1]))
    return inter/(a+b-inter)


def gt_box(annotation):
    x,y,w,h=annotation['bbox'];return [x,y,x+w,y+h]


def inside_site(predbox,truthbox):
    x,y=center(predbox);a,b,c,d=truthbox
    # A predicted lesion box may be much smaller than DENTEX's whole-tooth box.
    # This is a point-in-site metric, not box IoU or mAP.
    return a<=x<=c and b<=y<=d


def precision_recall(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,
            'precision':round(tp/(tp+fp),4) if tp+fp else None,
            'recall':round(tp/(tp+fn),4) if tp+fn else None}


def evaluate():
    zippath,data=download()
    by_image=defaultdict(list)
    for a in data['annotations']:by_image[a['image_id']].append(a)
    vision=Vision();findings={name:Counter() for name in ('yolo26','liodon')}
    tooth_counts=Counter();per_image=[]
    for index,(image,path) in enumerate(images(zippath,data['images']),1):
        gt=by_image[image['id']]
        # DENTEX validation labels cover abnormal teeth only; healthy teeth are not exhaustive GT.
        unique_teeth={((a['category_id_1']+1)*10+a['category_id_2']+1,tuple(a['bbox'])):a for a in gt}
        tooth_truth=list(unique_teeth.values())
        predicted,run=vision.predict('enumeration',path)
        teeth=[Tooth(**x) for x in predicted]
        guides=estimate_guides(teeth,image['width'],image['height'])
        assign_fdi(teeth,image['width'],image['height'],guides,'standard')
        matched=set();matched_with_fdi=0
        for tooth in sorted(teeth,key=lambda t:t.score,reverse=True):
            candidates=[(iou(tooth.bbox,gt_box(a)),j,a) for j,a in enumerate(tooth_truth) if j not in matched]
            if not candidates:continue
            score,j,a=max(candidates,key=lambda x:x[0])
            if score>=.3:
                matched.add(j)
                exact=(a['category_id_1']+1)*10+a['category_id_2']+1
                matched_with_fdi+=tooth.fdi==exact
        tooth_counts['annotated_abnormal_teeth']+=len(tooth_truth)
        tooth_counts['localized_iou_0_3']+=len(matched)
        tooth_counts['exact_fdi_after_localization']+=matched_with_fdi
        image_result={'file':image['file_name'],'annotated_abnormal_teeth':len(tooth_truth),
                      'tooth_localized':len(matched),'fdi_correct':matched_with_fdi,'findings':{}}
        for model_name in ('yolo26','liodon'):
            preds,model_run=vision.predict(model_name,path,include_all=model_name=='liodon')
            pred=[Finding(**p) for p in preds]
            categories=CATEGORIES if model_name=='liodon' else ('suspected_caries','periapical_radiolucency')
            image_result['findings'][model_name]={}
            for category in categories:
                sites=[a for a in gt if CATEGORY[a['category_id_3']]==category]
                candidates=[p for p in pred if p.label==category]
                used=set();tp=0
                for p in sorted(candidates,key=lambda x:x.score,reverse=True):
                    choices=[j for j,a in enumerate(sites) if j not in used and inside_site(p.bbox,gt_box(a))]
                    if choices:
                        j=min(choices,key=lambda j:abs(center(p.bbox)[0]-center(gt_box(sites[j]))[0]))
                        used.add(j);tp+=1
                fp=len(candidates)-tp;fn=len(sites)-tp
                counts=findings[model_name];counts[(category,'tp')]+=tp;counts[(category,'fp')]+=fp;counts[(category,'fn')]+=fn
                image_result['findings'][model_name][category]={'tp':tp,'fp':fp,'fn':fn}
        per_image.append(image_result)
        if index%10==0:print(f'Processed {index}/{len(data["images"])} panoramas',flush=True)
    metrics={name:{category:precision_recall(*(counter[(category,k)] for k in ('tp','fp','fn')))
                   for category in CATEGORIES if any(counter[(category,k)] for k in ('tp','fp','fn'))}
             for name,counter in findings.items()}
    denominator=tooth_counts['annotated_abnormal_teeth']
    result={'dataset':'DENTEX validation (50 OPGs, abnormal tooth annotations)',
            'dataset_url':'https://huggingface.co/datasets/ibrahimhamamci/DENTEX',
            'dataset_revision':REV,'archive_sha256':ZIP_SHA,'annotation_sha256':ANNOTATION_SHA,
            'metrics_definition':{
                'tooth_localization':'One-to-one greedy match of predicted tooth boxes to annotated abnormal tooth boxes, IoU >= 0.3. Healthy teeth have no exhaustive labels here.',
                'tooth_fdi':'Exact FDI from tooth type plus geometric quadrant among matched abnormal tooth boxes; no manual orientation correction.',
                'site_hit':'One-to-one class-matched predicted box-center inside annotated whole-tooth box. Caries and deep caries are merged. This is not object-detection mAP.',
                'thresholds':'Checkpoint-published settings: enumeration 0.25/640; YOLO26 0.348/1280; Liodon 0.45/640.',
                'limitations':'Small 50-image split, no patient-overlap audit, annotation geometry mismatch, model selection on public data possible, no clinical validation. Liodon was originally selected using DENTEX validation, so these are not independent Liodon performance estimates.'},
            'tooth':{**tooth_counts,
                'localization_recall':round(tooth_counts['localized_iou_0_3']/denominator,4),
                'exact_fdi_recall':round(tooth_counts['exact_fdi_after_localization']/denominator,4),
                'conditional_fdi_accuracy':round(tooth_counts['exact_fdi_after_localization']/tooth_counts['localized_iou_0_3'],4) if tooth_counts['localized_iou_0_3'] else None},
            'findings':metrics,'per_image':per_image}
    output=ROOT/'research/dentex_validation.json';output.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    result=evaluate()
    print(json.dumps({'tooth':result['tooth'],'findings':result['findings']},indent=2))
