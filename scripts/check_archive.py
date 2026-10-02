from pathlib import Path
import zipfile,hashlib
from huggingface_hub import HfApi
base=Path('/workspace/data/hf-cache/models--ShandaAI--FloodDiffusionDownloads/snapshots/97fe404e1b52293b6c8db47619fa7ab9b5459405')
p=base/'HumanML3D.zip'
info=HfApi().model_info('ShandaAI/FloodDiffusionDownloads',revision=base.name,files_metadata=True)
record=next(x for x in info.siblings if x.rfilename==p.name)
print('publisher_lfs',record.lfs,flush=True)
print('actual_sha256',hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),flush=True)
with zipfile.ZipFile(p) as z:
    for name in ['HumanML3D/texts/013202.txt','HumanML3D/train.txt','HumanML3D/Mean.npy']:
        try:
            content=z.read(name)
            print('OK',name,len(content),flush=True)
        except Exception as e: print(type(e).__name__,str(e),flush=True)
