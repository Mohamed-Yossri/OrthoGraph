import hashlib
import json
import os
import threading
import time
from pathlib import Path
from PIL import Image
from domain import Tooth, Finding, LABELS

ROOT = Path(__file__).resolve().parent
(ROOT/'.ultralytics').mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')


class Vision:
    def __init__(self):
        self.models = {}
        self.lock = threading.RLock()
        self.manifest = json.loads((ROOT/'research/assets.json').read_text())

    def load(self, name):
        if name not in self.models:
            from ultralytics import YOLO
            asset = self.manifest[name]
            path = ROOT/asset['path']
            if not path.exists():
                raise RuntimeError(f'Missing {name} weights. Run python download_assets.py.')
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
                raise RuntimeError(f'{name} checkpoint hash mismatch.')
            self.models[name] = YOLO(str(path))
        return self.models[name]

    def predict(self, name, image_path, include_all=False):
        import torch
        with self.lock:
            model = self.load(name)
            device = 0 if torch.cuda.is_available() else 'cpu'
            settings = {'enumeration':(640,.25), 'yolo26':(1280,.348), 'liodon':(640,.45)}
            size, conf = settings[name]
            started = time.perf_counter()
            result = model.predict(str(image_path), imgsz=size, conf=conf, device=device,
                                   verbose=False, **({'iou':.35} if name=='liodon' else {}))[0]
            if device == 0:
                torch.cuda.synchronize()
            elapsed = round((time.perf_counter()-started)*1000,1)
            records = []
            for i,box in enumerate(result.boxes):
                label = model.names[int(box.cls.item())]
                common = dict(bbox=[round(v,2) for v in box.xyxy[0].tolist()], score=round(box.conf.item(),4))
                if name == 'enumeration':
                    records.append(Tooth(id=f't{i+1:02}', tooth_type=int(label), **common).model_dump())
                elif label in LABELS and (name != 'liodon' or label == 'impacted_tooth' or include_all):
                    normalized,title,category = LABELS[label]
                    records.append(Finding(id=f'{name}-f{i+1:02}',label=normalized,title=title,
                                           category=category,raw_label=label,model=name,**common).model_dump())
            run = {'id':name,'model':name,'sha256':self.manifest[name]['sha256'],
                   'source':self.manifest[name]['url'],'imgsz':size,'threshold':conf,
                   'device':torch.cuda.get_device_name(0) if device==0 else 'cpu',
                   'predict_ms':elapsed,'detections':len(records)}
            return records,run


def crop_box(image_path, bbox, output=None):
    with Image.open(image_path) as img:
        x1,y1,x2,y2 = bbox
        padx,pady = max(30,(x2-x1)*.6),max(40,(y2-y1)*.45)
        box = (max(0,int(x1-padx)),max(0,int(y1-pady)),min(img.width,int(x2+padx)),min(img.height,int(y2+pady)))
        crop = img.crop(box)
        if output:
            crop.save(output)
        return crop.copy(), box
