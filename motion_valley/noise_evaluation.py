"""OF005: frozen OF004 checkpoint, independent and matched noise ensembles."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .overfit import approval_check, reconstruction_metrics
from .multi_overfit import make_controls, passes_reconstruction
from .model_factory import build_control_model
from .representation266 import Normalizer266, xyz266
from .flow_features import rollout
from .noise_statistics import sample_seed


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    cfg_path = Path('configs/noise_evaluation_v1.json')
    approval = approval_check('approvals/of005.json', cfg_path)
    cfg = json.loads(cfg_path.read_text())
    out = Path('/workspace/outputs/OF005')
    if out.exists(): raise FileExistsError('Never overwrite or restart OF005')
    deadline = datetime.datetime.fromisoformat(os.environ['EXPERIMENT_DEADLINE_UTC'].replace('Z', '+00:00')).timestamp()
    if deadline-time.time() < 300: raise RuntimeError('Insufficient allocation time')
    if not torch.cuda.is_available(): raise RuntimeError('GPU evaluation requires CUDA')
    if sha256(cfg['checkpoint']) != cfg['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    if sha256(cfg['model_config']) != cfg['model_config_sha256']: raise ValueError('Model configuration mismatch')
    if sha256(cfg['bank_file']) != cfg['bank_sha256']: raise ValueError('State bank mismatch')
    # Trusted task-produced checkpoint, identity verified above; optimizer is never used.
    state = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    model_cfg = json.loads(Path(cfg['model_config']).read_text())
    if state['config'] != model_cfg or state['step'] != 4000: raise ValueError('Unexpected checkpoint state')
    device = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
    model = build_control_model(model_cfg)
    model.load_state_dict(state['model'], strict=True)
    normalizer = Normalizer266(state['normalizer']['mean'], state['normalizer']['std']).to(device)
    del state
    model = model.to(device).eval().requires_grad_(False)
    with np.load(cfg['bank_file']) as bank:
        states = torch.from_numpy(bank['states'][0]).to(device)
        bank_angles = bank['angles'].tolist()
    out.mkdir(parents=True); (out/'samples').mkdir()
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg, model_config=model_cfg, approval=approval,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.parameters()),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
    valid = torch.ones(1, 64, dtype=torch.bool, device=device)
    start = time.monotonic(); total = 0; passed = 0
    with (out/'evaluation.jsonl').open('w', buffering=1) as log:
        for protocol in cfg['protocols']:
            for li, layout in enumerate(cfg['layouts']):
                for ai, angle in enumerate(cfg['angles']):
                    raw = states[bank_angles.index(angle)][None]
                    clean = normalizer.normalize(raw)
                    controls = make_controls(raw, normalizer, model_cfg['layouts'][layout])
                    generated_samples = []; seeds = []
                    def check(k, x):
                        if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite sample')
                        if not torch.equal(x[controls.known], controls.values[controls.known]):
                            raise AssertionError('Clean anchor changed')
                    for i in range(cfg['samples_per_condition']):
                        if deadline-time.time() < 120: raise RuntimeError('Deadline reserve reached; no automatic restart')
                        seed = sample_seed(protocol, li, ai, i)
                        noise = torch.randn(clean.shape, device=device,
                            generator=torch.Generator(device=device).manual_seed(seed))
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            generated = rollout(model, noise, controls, None, None, valid,
                                                model_cfg['sampling_steps'], model_cfg['mode'], callback=check)
                        physical = normalizer.denormalize(generated.float())
                        metrics = reconstruction_metrics(generated.float(), clean, normalizer, controls.known)
                        metrics['anchor_max_m'] = (xyz266(physical)-xyz266(raw)).norm(dim=-1)[controls.joints].max().item()
                        metrics.update(protocol=protocol, layout=layout, angle=angle, sample_index=i, seed=seed,
                                       all_steps_known_exact=True)
                        metrics['reconstruction_passed'] = passes_reconstruction(metrics, model_cfg['overfit_gate'])
                        passed += int(metrics['reconstruction_passed']); total += 1
                        log.write(json.dumps(metrics)+'\n')
                        generated_samples.append(physical[0].cpu().numpy()); seeds.append(seed)
                    np.savez_compressed(out/'samples'/f'{protocol}_{layout}_{angle}.npz',
                        generated266=np.stack(generated_samples), reference266=raw[0].cpu().numpy(),
                        known=controls.known[0].cpu().numpy(), seeds=seeds)
                    print(json.dumps(dict(protocol=protocol, layout=layout, angle=angle, completed=total,
                        elapsed_s=time.monotonic()-start, peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)), flush=True)
    assert total == 640
    summary = dict(completed_cases=total, reconstruction_pass_count=passed, elapsed_s=time.monotonic()-start,
                   peak_memory_gb=torch.cuda.max_memory_allocated()/1e9, checkpoint_unchanged=True)
    # Confirm the read-only evaluation never altered the saved weights.
    if sha256(cfg['checkpoint']) != cfg['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'COMPLETED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
