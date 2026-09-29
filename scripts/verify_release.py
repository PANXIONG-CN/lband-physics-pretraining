"""Verify the delivered release checksums without installing scientific libraries."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

def verify(root: Path, manifest: Path) -> int:
    count=0
    with manifest.open(encoding='utf-8-sig',newline='') as h:
        for row in csv.DictReader(h):
            p=(root/row['path']).resolve()
            if not p.is_relative_to(root.resolve()): raise ValueError('Unsafe manifest path')
            if not p.is_file(): raise FileNotFoundError(p)
            if p.stat().st_size!=int(row['bytes']): raise ValueError(f'Size mismatch: {row["path"]}')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError(f'Checksum mismatch: {row["path"]}')
            count+=1
    return count

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--public-only',action='store_true')
    a=ap.parse_args(); root=a.root.resolve()
    if a.public_only:
        forbidden=[p for p in root.rglob('*') if p.is_file() and
          (p.suffix.lower() in {'.tex','.bib','.otf','.ttf','.pem','.key'} or
           'manuscript' in p.parts or p.name.startswith('.env') or p.name=='.netrc')]
        if forbidden: raise ValueError('Non-public files detected: '+', '.join(str(p.relative_to(root)) for p in forbidden))
    counts={'release_files':verify(root,root/'release_manifest.csv'),
            'reproducibility_files':verify(root/'reproducibility',root/'reproducibility/sha256_manifest.csv')}
    print(json.dumps({'status':'PASS',**counts},indent=2))
if __name__=='__main__':main()
