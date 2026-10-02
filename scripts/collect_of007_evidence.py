"""Package only OF007 evidence after the once-only suite completes."""
import hashlib,json,tarfile
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
root=Path('/workspace/outputs');out=root/'OF007'
assert (out/'SUITE_COMPLETED').exists()
for name in ['baseline','density','data','coverage']:
    assert (out/(name+'_eval')/'COMPLETED').exists()
    assert len((out/(name+'_eval')/'evaluation.jsonl').read_text().splitlines())==160
checkpoints={}
for name,n in [('density',40),('data',320),('coverage',40)]:
    summary=json.loads((out/name/'summary.json').read_text())
    assert summary['completed_steps']==4000 and summary['actual_cases']==n and summary['completed_evaluation']
    p=out/name/'last.pt';checkpoints[name]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
files=[p for p in out.rglob('*') if p.is_file() and p.suffix not in ['.pt','.tmp']]
files+=[root/'OF007-run.log',root/'OF007-pip_freeze.txt',root/'OF007-source.tar.gz']
manifest=dict(checkpoints=checkpoints,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)})
index=root/'OF007-evidence-manifest.json';index.write_text(json.dumps(manifest,indent=2))
archive=root/'OF007-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as z:
    for p in files+[index]:z.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),checkpoints=checkpoints,files=len(files))))
