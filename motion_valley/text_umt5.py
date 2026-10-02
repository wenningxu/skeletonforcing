"""Reuse FloodDiffusion's actual UMT5-XXL implementation, weights and tokenizer."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
ENCODER_DIR='t5_umt5-xxl-enc-bf16'
WEIGHTS='models_t5_umt5-xxl-enc-bf16.pth'
UPSTREAM_REV='b611d6c4e24357a90066034935ed445179723b37'


def caption_key(caption): return hashlib.sha256(caption.encode('utf-8')).hexdigest()


def cache_texts(captions,deps,output,device='cuda'):
    sys.path.insert(0,str(ROOT/'third_party/FloodDiffusion'))
    from models.tools.t5 import T5EncoderModel
    base=Path(deps)/ENCODER_DIR; weights=base/WEIGHTS; tokenizer=base/'google/umt5-xxl'
    if not weights.is_file() or not tokenizer.is_dir(): raise FileNotFoundError('Official UMT5 weights/tokenizer missing; do not substitute another encoder')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with weights.open('rb') as f: weight_hash=hashlib.file_digest(f,'sha256').hexdigest()
    tokenizer_hashes={str(p.relative_to(tokenizer)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(tokenizer.rglob('*')) if p.is_file()}
    metadata=dict(encoder='FloodDiffusion/UMT5-XXL',upstream_revision=UPSTREAM_REV,
                  weights_sha256=weight_hash,tokenizer_sha256=tokenizer_hashes,text_len=512,
                  dimension=4096,dtype='bfloat16',pooling='none',clean='whitespace',
                  torch_version=torch.__version__)
    identity=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    # Cache is a frozen encoder output, not a trainable substitute or mean pool.
    encoder=T5EncoderModel(text_len=512,dtype=torch.bfloat16,device=torch.device(device),
                           checkpoint_path=str(weights),tokenizer_path=str(tokenizer))
    with torch.inference_mode():
        for caption in dict.fromkeys(captions):
            path=output/(caption_key(caption)+'.pt')
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=True)
                if old.get('identity')==identity and old.get('caption')==caption: continue
            context=encoder([caption],torch.device(device))[0].cpu()
            if context.ndim!=2 or context.shape[1]!=4096 or not torch.isfinite(context).all(): raise ValueError('invalid UMT5 context')
            torch.save(dict(caption=caption,identity=identity,context=context),path)
    (output/'manifest.json').write_text(json.dumps(dict(identity=identity,metadata=metadata),indent=2))


def read_cache(caption,folder):
    folder=Path(folder); manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['metadata']['encoder']!='FloodDiffusion/UMT5-XXL' or manifest['metadata']['pooling']!='none': raise ValueError('wrong text cache')
    item=torch.load(folder/(caption_key(caption)+'.pt'),map_location='cpu',weights_only=True)
    if item['caption']!=caption or item['identity']!=manifest['identity']: raise ValueError('stale text cache')
    context=item['context']
    if context.ndim!=2 or context.shape[-1]!=4096 or not torch.isfinite(context).all(): raise ValueError('invalid UMT5 cache')
    return context
