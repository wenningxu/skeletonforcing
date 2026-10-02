"""Local CPU integration audit of the full model recipe on one real 64-frame clip."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_valley.model_factory import build_control_model
from motion_valley.representation266 import position_mask
from motion_valley.flow_features import FeatureControls,teacher_state


def main():
    torch.set_num_threads(4); torch.manual_seed(1234)
    cfg=json.loads((ROOT/'configs/model_wan266_v1.json').read_text())
    model=build_control_model(cfg).cpu().eval()
    raw=torch.from_numpy(np.load(ROOT/'prepared/OF003/states.npz')['states'][:1,2])
    anchors=torch.zeros(1,64,22,dtype=torch.bool)
    for t,j in [(16,0),(32,20),(48,21)]: anchors[:,t,j]=True
    known=position_mask(anchors)
    controls=FeatureControls(torch.where(known,raw,0),known,anchors)
    x,sigma,_=teacher_state(raw,torch.randn_like(raw),12,32,controls)
    # Synthetic tokens test tensor plumbing, NOT text semantics or real model quality.
    text=torch.randn(1,12,4096); text_valid=torch.ones(1,12,dtype=torch.bool)
    valid=torch.ones(1,64,dtype=torch.bool); trace=[]
    def trace_layer(module,args):
        trace.append(dict(x_shape=list(args[0].shape),modulation_shape=list(args[1].shape),
                          modulation_dtype=str(args[1].dtype)))
    handles=[m.register_forward_pre_hook(trace_layer) for m in model.blocks]
    start=time.perf_counter()
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        result=model(x,sigma,known,text,text_valid,valid)
    elapsed=time.perf_counter()-start
    for handle in handles: handle.remove()
    assert result.shape==(1,64,266) and torch.isfinite(result).all()
    assert torch.count_nonzero(result)==0  # Official zero-head initialization, not learned output.
    parameter_count=sum(p.numel() for p in model.parameters())
    cfg['text_enabled']=False; torch.manual_seed(1234); no_text=build_control_model(cfg)
    main_cfg=json.loads((ROOT/'configs/model_wan266_main_v1.json').read_text())
    with torch.device('meta'):
        main_model=build_control_model(main_cfg)
    main_parameters=sum(p.numel() for p in main_model.parameters())
    files=['motion_valley/wan_components.py','motion_valley/model_wan266.py',
           'motion_valley/model_factory.py','tests/test_wan_standard.py','configs/model_wan266_v1.json',
           'configs/model_wan266_main_v1.json','motion_valley/control_experiment.py']
    record=dict(device='cpu',torch=torch.__version__,real_motion_input=True,text_tokens='synthetic shape-check only',
        trained_model=False,paid_gpu_run=False,parameters_text=parameter_count,
        parameters_no_text=sum(p.numel() for p in no_text.parameters()),trace=trace,
        main_size_parameters_text=main_parameters,main_size_validation='meta construction only; not executed',
        output_shape=list(result.shape),output_finite=True,zero_initialized_head_confirmed=True,
        single_cpu_forward_seconds=elapsed,not_a_streaming_latency_benchmark=True,
        hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    destination=ROOT/'reports/local_checks_wan_v1.json'
    destination.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
