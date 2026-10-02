"""Record the user's report approval and make an explicit, secret-free bundle."""
import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
approval_path=root/'approvals/of008-kimodo266.json';approval=json.loads(approval_path.read_text())
assert sha(root/'configs/kimodo266_overfit_v1.json')==approval['config_sha256']
assert sha(root/approval['report_file'])==approval['report_sha256']
checks=json.loads((root/'reports/local_checks_kimodo266.json').read_text())
for name,digest in checks['local_files'].items():assert sha(root/name)==digest,name
for name,digest in checks['official_files'].items():assert sha(root/'third_party/kimodo/kimodo'/name)==digest,name
approval.update(approved=True,user_approval_reference='2026-09-26 user explicitly approved OF008 report v9: 通过. One 4000-step Kimodo266 run, 80 routine + 160 common evaluations; 90-minute allocation and USD2 cap.')
approval_path.write_text(json.dumps(approval,indent=2)+'\n')
paths=[]
for folder in ['motion_valley','scripts','tests','configs','docs','approvals','prepared','third_party']:
    for p in (root/folder).rglob('*'):
        if not p.is_file() or any(s in p.parts for s in ['.git','__pycache__','.secrets']):continue
        if 'kimodo' in p.relative_to(root/'third_party').parts if folder=='third_party' else False:
            rel=str(p.relative_to(root/'third_party/kimodo')).replace('\\','/')
            if rel not in ['LICENSE','README.md']+['kimodo/'+x for x in checks['official_files']]:continue
        paths.append(p)
paths += [root/'requirements.txt']+list((root/'reports').glob('approval_report*_zh.md'))
paths += [root/'reports'/n for n in ['local_checks_kimodo266.json','kimodo_upstream.json','kimodo_official_config.yaml']]
archive=root/'outputs/OF008-source.tar.gz'
with tarfile.open(archive,'w:gz') as z:
    for p in sorted(set(paths)):
        rel=p.relative_to(root)
        assert not any(s in rel.parts for s in ['.secrets','.git','.venv'])
        z.add(p,arcname=str(rel))
record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(set(paths)),approval=approval)
(root/'reports/of008_upload_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
