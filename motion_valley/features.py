"""Deterministic evaluator adapter; never changes the generated XYZ artifact."""
import ast
from pathlib import Path
import sys
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def _load_reference():
    root=ROOT/'third_party/HumanML3D'
    sys.path.insert(0,str(root))
    source=(root/'notebook_reference.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
    module=types.ModuleType('humanml_feature_reference')
    exec(compile(tree,str(root/'notebook_reference.py'),'exec'),module.__dict__)
    module.n_raw_offsets=torch.from_numpy(module.t2m_raw_offsets)
    module.kinematic_chain=module.t2m_kinematic_chain
    module.face_joint_indx=[2,1,17,16]
    module.fid_r=[8,11]; module.fid_l=[7,10]
    # Generated XYZ already uses the HumanML skeleton. Retargeting would conceal
    # bone-length errors and move hard anchors, so omit that dataset preprocessing.
    module.uniform_skeleton=lambda positions,target: positions.copy()
    module.tgt_offsets=None
    return module


_reference=None


def xyz_to_features(xyz):
    global _reference
    if _reference is None: _reference=_load_reference()
    x=np.asarray(xyz,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(22,3) or len(x)<5 or not np.isfinite(x).all():
        raise ValueError('Expected finite T,22,3 XYZ, T>=5')
    features,canonical,_,_=_reference.process_file(x.copy(),.002)
    if features.shape != (len(x)-1,263) or not np.isfinite(features).all():
        raise ValueError('Degenerate skeleton produced invalid evaluator features')
    return features.astype(np.float32),canonical.astype(np.float32)
