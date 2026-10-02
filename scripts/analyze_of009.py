"""Verify downloaded hashes and independently audit all eight OF009 samples."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
import torch
from motion_valley.root_first import RootFirstSchedule
from motion_valley.representation266 import Normalizer266
from motion_valley.overfit import reconstruction_metrics
from motion_valley.multi_overfit import passes_reconstruction

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
collection=json.loads((root/'reports/of009_collection.json').read_text())
archive=root/'outputs/OF009-evidence.tar.gz';assert sha(archive)==collection['sha256']
with tarfile.open(archive) as z:
    assert all(m.name.startswith('OF009') and '..' not in Path(m.name).parts for m in z.getmembers())
    z.extractall(root/'outputs',filter='data')
manifest=json.loads((root/'outputs/OF009-evidence-manifest.json').read_text())
for name,e in manifest['files'].items():
    p=root/'outputs'/name;assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],name
resource_path=root/'reports/resources_of009.json';resource=json.loads(resource_path.read_text())
assert sha(root/'outputs/OF009-source.tar.gz')==resource['source_archive_sha256']
resource.update(evidence_sha256=collection['sha256'],evidence_files_verified=len(manifest['files']),
    checkpoint=manifest['checkpoint'],local_evidence='outputs/OF009-evidence.tar.gz')
out=root/'outputs/OF009';cfg=json.loads((root/'configs/root_first_overfit_v1.json').read_text())
if not manifest['completed']:
    resource.update(live_stage='Stopped run; partial evidence hashes verified',completed=False)
    resource_path.write_text(json.dumps(resource,indent=2)+'\n')
    print(json.dumps(dict(completed=False,verified_files=len(manifest['files']))));sys.exit(0)
assert json.loads((out/'manifest.json').read_text())['config']==cfg
summary=json.loads((out/'summary.json').read_text());assert summary['completed_steps']==4000 and summary['cases']==8
with np.load(out/'normalizer.npz') as d:norm=Normalizer266(torch.from_numpy(d['mean']),torch.from_numpy(d['std']))
with np.load(root/cfg['bank_file']) as d:gt=d['states'][cfg['bank_clip_index'],d['angles'].tolist().index(0)][None]
schedule=RootFirstSchedule(**cfg['schedule']);table=schedule.lattice(64).numpy()
with np.load(out/'time_field.npz') as d:np.testing.assert_array_equal(d['sigma'],table)
assert ((table[1:]<table[:-1]).sum(0)==32).all()
coverage=np.load(out/'training_step_coverage.npy');assert coverage.sum()==4000 and coverage.max()-coverage.min()<=1
rows=json.loads((out/'evaluation.json').read_text());audits=[]
for row,seed in zip(rows,cfg['heldout_noise_seeds']):
    assert row['seed']==seed
    with np.load(out/'samples'/f'{seed}.npz') as d:
        raw=d['generated266'];ref=d['reference266'];snaps=d['trajectory_generated266'];ks=d['snapshot_steps']
        np.testing.assert_array_equal(ref,gt);assert np.isfinite(raw).all()
        np.testing.assert_array_equal(snaps[-1],raw)
        for i in range(1,len(ks)):
            frozen=(table[ks[i-1]]==0)[None];np.testing.assert_array_equal(snaps[i][frozen],snaps[i-1][frozen])
    generated=torch.from_numpy(raw);truth=torch.from_numpy(ref)
    metrics=reconstruction_metrics(norm.normalize(generated),norm.normalize(truth),norm,torch.zeros_like(generated,dtype=torch.bool))
    for k,v in metrics.items():
        if k=='anchor_max_m':continue
        np.testing.assert_allclose(v,row[k],rtol=1e-4,atol=1e-6,err_msg=k)
    x=raw[...,4:70].reshape(1,64,22,3);y=ref[...,4:70].reshape(1,64,22,3)
    ratio=float(np.sqrt(np.mean(np.sum((x-x.mean(1,keepdims=True))**2,-1))/np.mean(np.sum((y-y.mean(1,keepdims=True))**2,-1))))
    np.testing.assert_allclose(ratio,row['temporal_rms_ratio'],rtol=1e-5,atol=1e-7)
    passed=passes_reconstruction(metrics,cfg['overfit_gate']) and cfg['temporal_ratio_min']<=ratio<=cfg['temporal_ratio_max']
    assert passed==row['overfit_passed'];audits.append(dict(seed=seed,passed=passed,temporal_rms_ratio=ratio))
keys=['feature_mse','xyz_mpjpe_m','root_mpjpe_m','rot6d_normalized_mse','velocity_normalized_mse',
      'velocity_error_vs_static_ratio','fk_position_disagreement_m','gt_fk_position_disagreement_m','temporal_rms_ratio']
result=dict(completed_steps=4000,cases=len(rows),passed=sum(a['passed'] for a in audits),
    mean={k:float(np.mean([r[k] for r in rows])) for k in keys},
    range={k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in keys},
    by_depth={str(d):float(np.mean([r['by_depth'][str(d)]['xyz_mpjpe_m'] for r in rows])) for d in range(8)},
    by_frame_block={str(f):float(np.mean([r['by_frame_block'][str(f)]['xyz_mpjpe_m'] for r in rows])) for f in [0,16,32,48]},
    independent_sample_audits=audits,training_step_coverage_min=int(coverage.min()),training_step_coverage_max=int(coverage.max()))
dest=root/'outputs/OF009-analysis';dest.mkdir(exist_ok=True);(dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
resource.update(completed=True,completed_at=(out/'COMPLETED').read_text(),live_stage='Completed; all evidence hashes and eight sample metrics audited locally')
resource_path.write_text(json.dumps(resource,indent=2)+'\n');print(json.dumps(result,indent=2))
