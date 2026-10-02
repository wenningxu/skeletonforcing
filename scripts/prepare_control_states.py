"""Build a small private diagnostic bank from verified saved GT; local CPU only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json
import numpy as np
import torch
from motion_valley.control_states import arm_variant,coherent_response
from motion_valley.diagnostics266 import consistency_metrics
from motion_valley.representation266 import position_mask,xyz266

ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(1)
    source=ROOT/'outputs/OF002-v3'; target=ROOT/'prepared/OF003'; target.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    raw=torch.from_numpy(np.concatenate([np.load(source/'samples'/f'{i:02d}_A_correct_None.npz')['reference266'] for i in range(8)]))
    angles=[-12,-6,0,6,12]; states=torch.stack([arm_variant(raw,x) for x in angles],1)
    layouts={'A':[[16,0],[32,20],[48,21]],'B':[[8,7],[44,21],[52,0]],
             'C':[[12,20],[52,19],[56,15]],'D':[[10,0],[46,19],[50,21]]}
    checks=[]
    for i,angle in enumerate(angles):
        m=consistency_metrics(states[:,i],states[:,i])
        original=consistency_metrics(states[:,i],raw)
        if m['fk_position_disagreement_m']>1e-5 or m['local_velocity_consistency_mae_m_frame']>1e-6 or original['bone_length_mae_m']>1e-5:
            raise ValueError('Kinematically inconsistent bank')
        checks.append(dict(angle=angle,consistency=m,bone_length_change_m=original['bone_length_mae_m']))
    minimum=1e9
    for layout in layouts.values():
        joints=torch.zeros(1,64,22,dtype=torch.bool)
        for t,j in layout: joints[:,t,j]=True
        known=position_mask(joints)
        for i in range(8):
            observed=states[i,:,known[0]]
            distances=torch.cdist(observed,observed); distances.fill_diagonal_(float('inf'))
            minimum=min(minimum,distances.min().item())
            for k in [0,1,3,4]:
                effect=coherent_response(states[i:i+1,k],states[i:i+1,2],states[i:i+1,k],states[i:i+1,2],known)
                if effect['response_target_rms_m']<.005: raise ValueError('Too little supervised control variation')
    if minimum<.005: raise ValueError('Control observations cannot distinguish variants reliably')
    path=target/'states.npz'
    np.savez_compressed(path,states=states.numpy(),source266=raw.numpy(),angles=np.array(angles))
    metadata=dict(source_experiment='OF002-v3',source_evidence_sha256='9cabbc7e030af652cff920cff65badd2851673ddacbce210d6711bf8aa971acf',
        articulation=dict(rotation_channel_joint=17,fk_pivot_joint=14,affected_joints=[17,19,21],axis='local_z',composition='R_original @ Rz',center_frame=48,radius_frames=12),
        selection=manifest['selection'],captions=manifest['captions'],angles=angles,train_angles=[-12,0,12],
        heldout_angles=[-6,6],layouts=layouts,minimum_variant_control_distance_m=minimum,checks=checks,
        zero_variant_max_xyz_change_m=(xyz266(states[:,2])-xyz266(raw)).abs().max().item(),
        bank_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),note='FK-consistent synthetic articulation, not collision-checked or new captured motion')
    (target/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(bank_sha256=metadata['bank_sha256'],shape=list(states.shape),minimum_control_distance_m=minimum,
                         zero_variant_change_m=metadata['zero_variant_max_xyz_change_m'],max_fk_m=max(x['consistency']['fk_position_disagreement_m'] for x in checks)),indent=2))

if __name__=='__main__': main()
