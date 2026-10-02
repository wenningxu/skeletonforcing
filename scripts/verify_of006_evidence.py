"""Verify downloaded evidence before releasing the task GPU."""
import argparse,hashlib,json,tarfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--sha256',required=True); args=parser.parse_args()
archive=root/'outputs/OF006-inpaint-evidence.tar.gz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)==args.sha256.lower()
with tarfile.open(archive) as z:
    assert all(m.name.startswith('OF006-inpaint') and '..' not in Path(m.name).parts for m in z.getmembers())
    z.extractall(root/'outputs',filter='data')
manifest=json.loads((root/'outputs/OF006-inpaint-evidence-manifest.json').read_text())
for name,entry in manifest['files'].items():
    p=root/'outputs'/name
    assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
source_sha=sha(root/'outputs/OF006-inpaint-source.tar.gz')
resource_path=root/'reports/resources_of006.json'; resource=json.loads(resource_path.read_text())
assert source_sha==resource['source_archive_sha256']
src=root/'outputs/OF006-inpaint'; cfg=json.loads((root/'configs/inpaint_baseline_v1.json').read_text())
assert json.loads((src/'manifest.json').read_text())['config']==cfg
assert (src/'SUITE_COMPLETED').exists()
assert len((src/'evaluation.jsonl').read_text().splitlines())==40
assert len((src/'independent.jsonl').read_text().splitlines())==160
resource.update(evidence_sha256=args.sha256.lower(),evidence_files_verified=len(manifest['files']),
    checkpoint=manifest['checkpoint'],local_evidence='outputs/OF006-inpaint-evidence.tar.gz',
    completed_at=(src/'SUITE_COMPLETED').read_text())
resource_path.write_text(json.dumps(resource,indent=2))
print(json.dumps(dict(verified_files=len(manifest['files']),source_sha256=source_sha,
    checkpoint=manifest['checkpoint'],complete=True),indent=2))
