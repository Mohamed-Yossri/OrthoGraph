"""Reproduce runtime checks; this is NOT a clinical accuracy evaluation."""
import gc
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent
(ROOT / ".ultralytics").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import torch
import ultralytics
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["enumeration", "liodon", "gegesay", "yolo26"])
    args = parser.parse_args()
    manifest = json.loads((ROOT / "research/assets.json").read_text())
    output = ROOT / "research/verification"
    output.mkdir(exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    report = {"purpose": "single-image runtime smoke test, not accuracy validation",
              "torch": torch.__version__, "ultralytics": ultralytics.__version__,
              "device": torch.cuda.get_device_name(0) if device == 0 else "cpu",
              "models": {}}
    if args.models and (output / "report.json").exists():
        report = json.loads((output / "report.json").read_text())
    settings = {"enumeration": (640, .25), "liodon": (640, .45),
                "gegesay": (640, .25), "yolo26": (1280, .348)}
    for name, (size, conf) in settings.items():
        if args.models and name not in args.models:
            continue
        item = manifest[name]
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        model = None
        try:
            if device == 0:
                torch.cuda.reset_peak_memory_stats()
            model = YOLO(str(path))
            kwargs = dict(source=str(ROOT / manifest["sample"]["path"]),
                          device=device, imgsz=size, conf=conf, verbose=False)
            if name == "liodon":
                kwargs["iou"] = .35
            model.predict(**kwargs)
            times = []
            for _ in range(5):
                if device == 0:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = model.predict(**kwargs)[0]
                if device == 0:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            result.save(filename=str(output / f"{name}.jpg"))
            detections = json.loads(result.to_json())
            (output / f"{name}.json").write_text(json.dumps(detections, indent=2))
            report["models"][name] = {
                "status": "passed", "task": model.task, "names": model.names,
                "imgsz": size, "conf": conf, "detections": len(result.boxes),
                "median_warm_predict_ms": statistics.median(times),
                "peak_allocated_mb": torch.cuda.max_memory_allocated()/1024**2 if device == 0 else None,
                "timing_scope": "5 sequential predictions, batch=1, includes preprocessing and postprocessing; excludes load, warmup and saving",
            }
        except Exception as exc:
            report["models"][name] = {"status": "failed", "error": str(exc)}
        finally:
            del model
            gc.collect()
            if device == 0:
                torch.cuda.empty_cache()
        print(name, json.dumps(report["models"][name]), flush=True)
        (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
