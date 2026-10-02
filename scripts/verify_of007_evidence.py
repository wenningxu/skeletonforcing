"""Verify hashes and completeness before releasing the OF007 pod."""
import argparse,hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--sha256',required=True);args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=root/'outputs/OF007-evidence.tar.gz';assert sha(archive)==args.sha256.lower()
with tarfile.open(archive) as z:
    assert all(m.name.startswith('OF007') and '..' not in Path(m.name).parts for m in z.getmembers())
    z.extractall(root/'outputs',filter='data')
manifest=json.loads((root/'outputs/OF007-evidence-manifest.json').read_text())
for name,entry in manifest['files'].items():
    p=root/'outputs'/name;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
rp=root/'reports/resources_of007.json';r=json.loads(rp.read_text())
assert sha(root/'outputs/OF007-source.tar.gz')==r['source_archive_sha256']
src=root/'outputs/OF007';plan=json.loads((root/'configs/control_factors_v1.json').read_text())
assert json.loads((src/'manifest.json').read_text())['plan']==plan and (src/'SUITE_COMPLETED').exists()
for name in ['baseline','density','data','coverage']:
    assert len((src/(name+'_eval')/'evaluation.jsonl').read_text().splitlines())==160
for item in plan['arms']:
    assert json.loads((src/item['name']/'manifest.json').read_text())['config']==json.loads((root/item['config']).read_text())
r.update(evidence_sha256=args.sha256.lower(),evidence_files_verified=len(manifest['files']),
    checkpoints=manifest['checkpoints'],local_evidence='outputs/OF007-evidence.tar.gz',completed_at=(src/'SUITE_COMPLETED').read_text())
rp.write_text(json.dumps(r,indent=2));print(json.dumps(dict(files_verified=len(manifest['files']),checkpoints=manifest['checkpoints'],complete=True),indent=2))
