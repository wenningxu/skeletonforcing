"""Official FloodDiffusion/HumanML3D frozen evaluators, with explicit XYZ adapter."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/FloodDiffusion'))
from metrics.tools.t2m_evaluator import MovementConvEncoder,MotionEncoderBiGRUCo,TextEncoderBiGRUCo
from metrics.tools.word_vectorizer import WordVectorizer
from metrics.tools.utils import calculate_activation_statistics_np,calculate_frechet_distance_np,calculate_diversity_np


class Evaluator:
    def __init__(self,data_root,device):
        self.device=device; root=Path(data_root)/'deps'
        self.mean=torch.tensor(np.load(root/'t2m/meta/mean.npy'),device=device).float()
        self.std=torch.tensor(np.load(root/'t2m/meta/std.npy'),device=device).float()
        if (self.std<=0).any(): raise ValueError('invalid evaluator normalization')
        self.movement=MovementConvEncoder(259,512,512)
        self.motion=MotionEncoderBiGRUCo(512,1024,512)
        self.text=TextEncoderBiGRUCo(300,15,512,512)
        for name,model in [('movement_encoder',self.movement),('motion_encoder',self.motion),('text_encoder',self.text)]:
            model.load_state_dict(torch.load(root/f't2m/humanml3d/{name}.pt',map_location='cpu',weights_only=True))
            model.to(device).eval().requires_grad_(False)
        self.words=WordVectorizer(str(root/'glove'),'our_vab')

    @torch.no_grad()
    def motions(self,features):
        lengths=np.array([len(x) for x in features]); order=np.argsort(-lengths).copy()
        x=torch.zeros(len(features),int(lengths.max()),263,device=self.device)
        for i,k in enumerate(order): x[i,:lengths[k]]=torch.as_tensor(features[k],device=self.device)
        emb=self.motion(self.movement(((x-self.mean)/self.std)[...,:-4]),torch.tensor(lengths[order]//4))
        return emb[torch.as_tensor(np.argsort(order),device=self.device)].cpu().numpy()

    @torch.no_grad()
    def texts(self,tokens_list):
        vectors=[]; positions=[]; lengths=[]
        for tokens in tokens_list:
            tokens=['sos/OTHER']+tokens[:20]+['eos/OTHER']; lengths.append(len(tokens))
            tokens+=['unk/OTHER']*(22-len(tokens))
            pairs=[self.words[t] for t in tokens]
            vectors.append(np.stack([x[0] for x in pairs])); positions.append(np.stack([x[1] for x in pairs]))
        out=self.text(torch.tensor(np.array(vectors),device=self.device).float(),
                      torch.tensor(np.array(positions),device=self.device).float(),torch.tensor(lengths))
        return out.cpu().numpy()


def embedding_metrics(generated,reference,text,seed=0):
    n=len(generated)
    if n<32: raise ValueError('at least 32 motions for retrieval groups')
    result={'embedding_sample_count':n,'retrieval_group_size':32,'diversity_pairs':min(300,n-1)}
    mu,cov=calculate_activation_statistics_np(generated)
    rmu,rcov=calculate_activation_statistics_np(reference)
    result['FID']=float(calculate_frechet_distance_np(rmu,rcov,mu,cov))
    np.random.seed(seed)
    result['Diversity']=float(calculate_diversity_np(generated,min(300,n-1)))
    np.random.seed(seed)
    result['gt_Diversity']=float(calculate_diversity_np(reference,min(300,n-1)))
    result['MM_Dist']=float(np.linalg.norm(generated-text,axis=-1).mean())
    result['gt_MM_Dist']=float(np.linalg.norm(reference-text,axis=-1).mean())
    permutation=np.random.default_rng(seed).permutation(n)
    for name,emb in [('',generated),('gt_',reference)]:
        counts=np.zeros(3); groups=n//32
        for i in range(groups):
            ids=permutation[i*32:(i+1)*32]
            d=np.linalg.norm(text[ids,None,:]-emb[None,ids,:],axis=-1)
            rank=np.argsort(d,axis=-1)
            for k in range(1,4): counts[k-1]+=np.any(rank[:,:k]==np.arange(32)[:,None],axis=1).sum()
        for k in range(1,4): result[f'{name}R_precision_top_{k}']=float(counts[k-1]/(groups*32))
    return result
