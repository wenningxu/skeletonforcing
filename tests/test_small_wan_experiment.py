import copy
import json
from pathlib import Path
import pytest
import torch
from motion_valley.control_experiment import select_bank_subset,summarize


def test_small_bank_selection_and_summary_are_not_hardcoded_to_eight():
    cfg=json.loads(Path('configs/control_no_text_v4.json').read_text())
    cfg.update(clips=1,bank_clip_indices=[2],batch_size=3,train_layouts=['A'],eval_layouts=['A','D'])
    states=torch.randn(8,5,64,266); meta={'captions':[str(i) for i in range(8)]}
    selected,captions,indices=select_bank_subset(states,meta,cfg)
    assert torch.equal(selected,states[2:3]) and captions==['2'] and indices==[2]
    summary=summarize([],cfg,False,0)
    assert summary['expected_cases']==40
    assert [summary[g]['expected'] for g in ['trained_layout_trained_angle','trained_layout_unseen_angle',
            'unseen_layout_trained_angle','unseen_layout_unseen_angle']]==[12,8,12,8]
    bad=copy.deepcopy(cfg); bad['bank_clip_indices']=[8]
    with pytest.raises(ValueError): select_bank_subset(states,meta,bad)
    bad=copy.deepcopy(cfg); bad.update(clips=2,bank_clip_indices=[2,2])
    with pytest.raises(ValueError): select_bank_subset(states,meta,bad)
    bad=copy.deepcopy(cfg); bad['batch_size']=4
    with pytest.raises(ValueError): select_bank_subset(states,meta,bad)
