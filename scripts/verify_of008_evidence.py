"""Verify downloaded OF008 archive and every evidence file before GPU teardown."""
import argparse,hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--sha256',required=True);args=p.parse_args()
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
archive=root/'outputs/OF008-evidence.tar.gz';assert sha(archive)==args.sha256
with tarfile.open(archive) as z:
    assert all(m.name.startswith('OF008') and '..' not in Path(m.name).parts for m in z.getmembers())
    z.extractall(root/'outputs',filter='data')
manifest=json.loads((root/'outputs/OF008-evidence-manifest.json').read_text())
for name,entry in manifest['files'].items():
    path=root/'outputs'/name
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],name
cfg=json.loads((root/'configs/kimodo266_overfit_v1.json').read_text())
src=root/'outputs/OF008'
assert json.loads((src/'train/manifest.json').read_text())['config']==cfg
summary=json.loads((src/'train/summary.json').read_text())
assert summary['completed_steps']==4000 and summary['actual_cases']==80 and summary['completed_evaluation']
assert (src/'SUITE_COMPLETED').exists() and (src/'common_eval/COMPLETED').exists()
assert len((src/'common_eval/evaluation.jsonl').read_text().splitlines())==160
rp=root/'reports/resources_of008.json';resource=json.loads(rp.read_text())
assert sha(root/'outputs/OF008-source.tar.gz')==resource['source_archive_sha256']
assert json.loads((src/'source_checks.json').read_text())==json.loads((root/'reports/local_checks_kimodo266.json').read_text())
resource.update(evidence_sha256=args.sha256,evidence_files_verified=len(manifest['files']),
    checkpoint=manifest['checkpoint'],local_evidence='outputs/OF008-evidence.tar.gz',
    completed_at=(src/'SUITE_COMPLETED').read_text(),live_stage='Completed; local evidence hashes verified')
rp.write_text(json.dumps(resource,indent=2)+'\n')
print(json.dumps(dict(complete=True,verified_files=len(manifest['files']),checkpoint=manifest['checkpoint']),indent=2))
