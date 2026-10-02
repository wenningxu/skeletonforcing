"""Run on pod after OF006-inpaint completion; never contains private data."""
import hashlib,json,tarfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

root=Path('/workspace/outputs'); out=root/'OF006-inpaint'
assert (out/'SUITE_COMPLETED').exists()
assert len((out/'evaluation.jsonl').read_text().splitlines())==40
assert len((out/'independent.jsonl').read_text().splitlines())==160
assert json.loads((out/'summary.json').read_text())['completed_steps']==4000
files=[p for p in out.rglob('*') if p.is_file() and p.suffix not in ['.pt','.tmp']]
files+=[root/'OF006-inpaint-run.log',root/'OF006-inpaint-pip_freeze.txt',root/'OF006-inpaint-source.tar.gz']
checkpoint=out/'last.pt'
manifest=dict(checkpoint=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)),
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
index=root/'OF006-inpaint-evidence-manifest.json';index.write_text(json.dumps(manifest,indent=2))
archive=root/'OF006-inpaint-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as z:
    for p in files+[index]:z.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
    checkpoint=manifest['checkpoint'],files=len(files)),indent=2))
