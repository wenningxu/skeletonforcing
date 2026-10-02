"""Collect OF009 evidence, including a stopped run; retain checkpoints on volume."""
import hashlib,json,tarfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

root=Path('/workspace/outputs');out=root/'OF009'
files=[p for p in out.rglob('*') if p.is_file() and p.suffix not in ['.pt','.tmp']]
files += [p for p in [root/'OF009-run.log',root/'OF009-pip_freeze.txt',root/'OF009-source.tar.gz'] if p.exists()]
files += [p for p in root.glob('OF009*setup*') if p.is_file()]
cp=out/'last.pt'
checkpoint=dict(path=str(cp),bytes=cp.stat().st_size,sha256=sha(cp)) if cp.exists() else None
manifest=dict(completed=(out/'COMPLETED').exists(),checkpoint=checkpoint,
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
index=root/'OF009-evidence-manifest.json';index.write_text(json.dumps(manifest,indent=2))
archive=root/'OF009-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as z:
    for p in files+[index]:z.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),checkpoint=checkpoint,files=len(files),completed=manifest['completed'])))
