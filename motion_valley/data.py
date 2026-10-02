import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/FloodDiffusion'))
from utils.motion_process import recover_joint_positions_263
from metrics.tools.word_vectorizer import WordVectorizer


def split_ids(root, split, babel=False):
    filename=split+'_processed.txt' if babel else split+'.txt'
    path=Path(root)/filename
    if not path.exists(): raise FileNotFoundError(path)
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def assert_disjoint(root,babel=False):
    sets={}
    for split in ['train','val','test']:
        try: ids=split_ids(root,split,babel)
        except FileNotFoundError: continue
        sets[split]=set(x[1:] if not babel and x.startswith('M') else x for x in ids)
    for a in sets:
        for b in sets:
            if a<b and sets[a]&sets[b]:
                raise ValueError(f'{root}: {a}/{b} overlap ({len(sets[a]&sets[b])})')
    return {k:len(v) for k,v in sets.items()}


class MotionDataset(Dataset):
    def __init__(self, root, split='train', frames=64, limit=0, dataset='HumanML3D'):
        self.root=Path(root); self.split=split; self.frames=frames
        self.babel=dataset=='BABEL_streamed'
        self.data_root=self.root/'raw_data'/dataset
        self.splits=assert_disjoint(self.data_root,self.babel)
        self.vectorizer=WordVectorizer(str(self.root/'deps/glove'),'our_vab')
        self.records=[]
        folder='motions' if self.babel else 'new_joint_vecs'
        # A deterministic XYZ cache uses only the motion itself, no fitted statistics.
        cache=self.root/'xyz_cache'/dataset; cache.mkdir(parents=True,exist_ok=True)
        rejected=[]
        for name in split_ids(self.data_root,split,self.babel):
            path=self.data_root/folder/(name+'.npy')
            textpath=self.data_root/'texts'/(name+'.txt')
            if not path.exists() or not textpath.exists():
                rejected.append([name,'missing']); continue
            feat=np.load(path)
            if feat.ndim!=2 or feat.shape[1]!=263 or not np.isfinite(feat).all():
                rejected.append([name,'invalid']); continue
            if len(feat)<40 or (not self.babel and len(feat)>200):
                rejected.append([name,'length']); continue
            cached=cache/(name+'.npy')
            if cached.exists(): xyz=np.load(cached)
            else:
                xyz=recover_joint_positions_263(feat.astype(np.float32),22)
                cached.parent.mkdir(parents=True,exist_ok=True); np.save(cached,xyz)
            texts=[]
            for line in textpath.read_text(encoding='utf-8').splitlines():
                parts=line.strip().split('#')
                if len(parts)!=4: continue
                caption,tokens,start,end=parts; start=float(start); end=float(end)
                start=0 if not np.isfinite(start) else start
                end=0 if not np.isfinite(end) else end
                tokens=tokens.split()
                emb=np.mean([self.vectorizer[x][0] for x in tokens],axis=0) if tokens else np.zeros(300)
                texts.append(dict(caption=caption,tokens=tokens,start=int(start*20),end=int(end*20),embedding=emb.astype(np.float32)))
            if not texts: rejected.append([name,'no text']); continue
            if self.babel:
                self.records.append(dict(name=name,xyz=xyz,feature=feat,texts=texts))
            else:
                full=[x for x in texts if x['start']==0 and x['end']==0]
                if full: self.records.append(dict(name=name,xyz=xyz,feature=feat,texts=full))
                for k,tx in enumerate(texts):
                    if tx['end']>tx['start']:
                        a=max(0,tx['start']); b=min(len(xyz),tx['end'])
                        if b-a>=40:
                            self.records.append(dict(name=f'{name}:segment{k}',xyz=xyz[a:b],feature=feat[a:b],texts=[tx]))
            if limit and len(self.records)>=limit: break
        if limit: self.records=self.records[:limit]
        self.audit=dict(dataset=dataset,split=split,splits=self.splits,records=len(self.records),rejected=rejected)
        if not self.records: raise ValueError(f'No records: {self.audit}')

    def __len__(self): return len(self.records)

    def __getitem__(self,i):
        r=self.records[i]; xyz=r['xyz']; n=len(xyz)
        length=min(n,self.frames)
        start=random.randint(0,n-length) if self.split=='train' else 0
        x=xyz[start:start+length].copy()
        # Crop origin translation applied equally to all joints; keep world Y.
        origin=x[0,0].copy(); origin[1]=0; x-=origin
        text=np.zeros((length,300),np.float32)
        tx=random.choice(r['texts']) if self.split=='train' else r['texts'][0]
        text[:]=tx['embedding']
        if self.babel:
            for segment in r['texts']:
                a=max(0,segment['start']-start); b=min(length,segment['end']-start)
                if b>a: text[a:b]=segment['embedding']
        return dict(x=torch.from_numpy(x),text=torch.from_numpy(text),length=length,
                    name=r['name'],tokens=tx['tokens'],feature=torch.from_numpy(r['feature'][start:start+length].astype(np.float32)))


def collate(batch):
    b=len(batch); t=max(r['length'] for r in batch)
    x=torch.zeros(b,t,22,3); text=torch.zeros(b,t,300); valid=torch.zeros(b,t,dtype=torch.bool)
    feat=torch.zeros(b,t,263)
    for i,r in enumerate(batch):
        n=r['length']; x[i,:n]=r['x']; text[i,:n]=r['text']; valid[i,:n]=True; feat[i,:n]=r['feature']
    return dict(x=x,text=text,valid=valid,feature=feat,lengths=[r['length'] for r in batch],
                names=[r['name'] for r in batch],tokens=[r['tokens'] for r in batch])


def random_controls(valid, generator=None, max_anchors=8, clean_prefix=False):
    b,t=valid.shape; mask=torch.zeros(b,t,22,dtype=torch.bool,device=valid.device)
    for i in range(b):
        length=int(valid[i].sum())
        count=int(torch.randint(1,max_anchors+1,(1,),generator=generator,device=valid.device))
        frames=torch.randint(length,(count,),generator=generator,device=valid.device)
        joints=torch.randint(22,(count,),generator=generator,device=valid.device)
        mask[i,frames,joints]=True
        if clean_prefix:
            prefix=int(torch.randint(0,min(9,length),(1,),generator=generator,device=valid.device))
            mask[i,:prefix]=True
    return mask
