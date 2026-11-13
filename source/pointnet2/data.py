from pathlib import Path
from collections import OrderedDict
import json,numpy as np,torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parent

class Blocks(Dataset):
    def __init__(self,split='train',points=4096,seed=42,limit=None,augment=True,stats_name='train'):
        self.blocks=json.loads((ROOT/f'{split}_blocks.json').read_text())
        if limit and len(self.blocks)>limit: self.blocks=[self.blocks[i] for i in np.linspace(0,len(self.blocks)-1,limit,dtype=int)]
        self.points=points; self.seed=seed; self.epoch=0; self.augment=augment and split.startswith('train'); self.cache=OrderedDict(); self.stats=json.loads((ROOT/f'{stats_name}_stats.json').read_text())
    def __len__(self): return len(self.blocks)
    def arrays(self,b):
        key=(b['sequence'],b['scan'],b['shift'])
        if key not in self.cache:
            prefix=ROOT/'cache'/b['sequence']/f"{b['scan']:06d}"
            self.cache[key]=(np.load(str(prefix)+'_x.npy',mmap_mode='r'),np.load(str(prefix)+'_y.npy',mmap_mode='r'),np.load(str(prefix)+f"_order{b['shift']}.npy",mmap_mode='r'))
            if len(self.cache)>4: self.cache.popitem(last=False)
        self.cache.move_to_end(key); return self.cache[key]
    def __getitem__(self,i):
        b=self.blocks[i]; x,y,order=self.arrays(b); rng=np.random.default_rng(self.seed+i*1009+self.epoch*1000003)
        ids=order[b['start']:b['stop']]
        if self.augment and rng.random()<.5:
            ids=rng.choice(ids,max(64,int(len(ids)*rng.uniform(.7,1))),replace=False)
        ids=rng.choice(ids,self.points,replace=len(ids)<self.points); feat=np.array(x[ids],copy=True); target=np.array(y[ids],np.int64)
        feat[:,0]-=b['cx']; feat[:,1]-=b['cy']; feat[:,3]=(feat[:,3]-self.stats['remission_mean'])/self.stats['remission_std']
        if self.augment:
            theta=rng.uniform(-np.pi,np.pi); c,s=np.cos(theta),np.sin(theta); feat[:,:2]=feat[:,:2]@np.array([[c,-s],[s,c]],np.float32)
            if rng.random()<.5: feat[:,0]*=-1
            feat[:,:3]*=rng.uniform(.95,1.05); feat[:,:3]+=np.clip(rng.normal(0,.01,(len(feat),3)),-.03,.03).astype(np.float32)
            feat[:,3]+=rng.normal(0,.02,len(feat)).astype(np.float32)
        return torch.from_numpy(feat.T.copy()),torch.from_numpy(target)

def class_weights(dataset='train'):
    counts=np.array(json.loads((ROOT/f'{dataset}_stats.json').read_text())['counts']); f=counts/max(counts.sum(),1)
    w=np.clip(1/np.sqrt(np.maximum(f,1e-4)),1,10); w[counts==0]=0; w/=w[w>0].mean()
    return w.astype(np.float32)
