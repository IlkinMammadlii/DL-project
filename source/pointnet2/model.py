"""PointNet++ SSG adapted from yanx27 (MIT; pinned source in vendor/).

Four FPS/radius-grouped set abstractions, four inverse-distance feature
propagation stages, and pointwise semantic logits. Geometry always FP32.
"""
import torch
from torch import nn
from vendor import pointnet2_utils as u

@torch.jit.script
def fps(xyz:torch.Tensor,npoint:int):
    b,n,_=xyz.shape
    out=torch.empty((b,npoint),dtype=torch.long,device=xyz.device)
    distance=torch.full((b,n),1e10,device=xyz.device)
    # Geometry-dependent initial point, deterministic for distinct positions.
    farthest=((xyz-xyz.mean(1,keepdim=True))**2).sum(-1).argmax(-1)
    batch=torch.arange(b,device=xyz.device)
    for i in range(npoint):
        out[:,i]=farthest
        centre=xyz[batch,farthest].unsqueeze(1)
        distance=torch.minimum(distance,((xyz-centre)**2).sum(-1))
        farthest=distance.argmax(-1)
    return out

def distance(src,dst):
    with torch.autocast(device_type=src.device.type,enabled=False):
        return torch.cdist(src.float(),dst.float(),p=2,compute_mode='donot_use_mm_for_euclid_dist').square()

def ball_query(radius,nsample,xyz,new_xyz):
    # Nearest K within the ball, padding with the nearest valid point.
    # Centroids are selected from xyz, so at least the centroid itself is valid.
    d=distance(new_xyz,xyz)
    vals,idx=d.topk(min(nsample,xyz.shape[1]),dim=-1,largest=False,sorted=True)
    idx=torch.where(vals<=radius*radius,idx,idx[:,:,:1])
    return idx

u.farthest_point_sample=fps
u.square_distance=distance
u.query_ball_point=ball_query

class PointNet2(nn.Module):
    def __init__(self,centroids=(512,128,32,8),radii=(.8,1.6,3.2,6.4),remission=True):
        super().__init__(); self.remission=remission
        self.sa1=u.PointNetSetAbstraction(centroids[0],radii[0],32,7 if remission else 6,[32,32,64],False)
        self.sa2=u.PointNetSetAbstraction(centroids[1],radii[1],32,67,[64,64,128],False)
        self.sa3=u.PointNetSetAbstraction(centroids[2],radii[2],32,131,[128,128,256],False)
        self.sa4=u.PointNetSetAbstraction(centroids[3],radii[3],32,259,[256,256,512],False)
        self.fp4=u.PointNetFeaturePropagation(768,[256,256])
        self.fp3=u.PointNetFeaturePropagation(384,[256,256])
        self.fp2=u.PointNetFeaturePropagation(320,[256,128])
        self.fp1=u.PointNetFeaturePropagation(132 if remission else 131,[128,128,128])
        self.head=nn.Sequential(nn.Conv1d(128,128,1),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.3),nn.Conv1d(128,19,1))
    def forward(self,x):
        xyz=x[:,:3].float()
        # Absolute height and local XY remain available via feature skip connections.
        feat=x.clone(); feat[:,:3]=feat[:,:3]/torch.tensor([8.,8.,5.],device=x.device).view(1,3,1)
        a,fa=self.sa1(xyz,feat); b,fb=self.sa2(a,fa); c,fc=self.sa3(b,fb); d,fd=self.sa4(c,fc)
        fc=self.fp4(c,d,fc,fd); fb=self.fp3(b,c,fb,fc); fa=self.fp2(a,b,fa,fb); out=self.fp1(xyz,a,feat,fa)
        return self.head(out)

if __name__=='__main__':
    import time,json
    torch.set_num_threads(2); torch.manual_seed(42); m=PointNet2().cuda(); opt=torch.optim.AdamW(m.parameters(),lr=.001)
    results=[]
    for batch in [2,4,8]:
        x=torch.randn(batch,4,4096,device='cuda'); x[:,:2]*=4; y=torch.randint(0,19,(batch,4096),device='cuda'); times=[]; torch.cuda.reset_peak_memory_stats()
        for i in range(8):
            torch.cuda.synchronize(); t=time.perf_counter(); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): loss=nn.functional.cross_entropy(m(x),y)
            loss.backward(); opt.step(); torch.cuda.synchronize(); times.append(time.perf_counter()-t)
        r={'batch':batch,'points':4096,'seconds_per_step':sum(times[3:])/len(times[3:]),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'parameters':sum(p.numel() for p in m.parameters())}; results.append(r); print(r,flush=True)
    from pathlib import Path
    (Path(__file__).parent/'benchmark.json').write_text(json.dumps(results,indent=2))
