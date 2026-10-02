"""Package completed OF008 evidence while retaining the checkpoint on the volume."""
import hashlib,json,tarfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

root=Path('/workspace/outputs');out=root/'OF008'
assert (out/'SUITE_COMPLETED').exists() and (out/'train/COMPLETED').exists()
summary=json.loads((out/'train/summary.json').read_text())
assert summary['completed_steps']==4000 and summary['actual_cases']==80 and summary['completed_evaluation']
assert (out/'common_eval/COMPLETED').exists()
assert len((out/'common_eval/evaluation.jsonl').read_text().splitlines())==160
checkpoint=out/'train/last.pt'
cp=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=sha(checkpoint))
files=[p for p in out.rglob('*') if p.is_file() and p.suffix not in ['.pt','.tmp']]
files += [root/'OF008-run.log',root/'OF008-pip_freeze.txt',root/'OF008-source.tar.gz']
manifest=dict(checkpoint=cp,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
index=root/'OF008-evidence-manifest.json';index.write_text(json.dumps(manifest,indent=2))
archive=root/'OF008-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as z:
    for p in files+[index]:z.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),checkpoint=cp,files=len(files))))
