"""Download pinned research assets and verify every file before use."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--demo',action='store_true',help='Include the upstream example for local research testing')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'research/assets.json').read_text())
    for name in ['enumeration','yolo26','liodon']+(['sample'] if args.demo else []):
        item=manifest[name];path=ROOT/item['path']
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:
            print(name,'verified');continue
        path.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(item['url'],timeout=120) as response:
            data=response.read()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise RuntimeError(f'{name}: hash mismatch; file not installed')
        path.write_bytes(data)
        print(name,'downloaded and verified')

if __name__=='__main__':
    main()
