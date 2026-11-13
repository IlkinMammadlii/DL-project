"""Targeted architecture, geometry, metric and full-scene coverage checks."""
import json,numpy as np,torch,yaml
from data import ROOT
from model import PointNet2,fps,ball_query
from train import scores
from infer import predict_scene
def main():
    torch.set_num_threads(2); torch.manual_seed(3); tests=[]
    xyz=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]]]); ids=fps(xyz,3); assert len(ids[0].unique())==3
    query=xyz[:,[0,2]]; groups=ball_query(1.01,4,xyz,query); got=xyz[0,groups[0]]; assert ((got-query[0,:,None])**2).sum(-1).max()<=1.01**2+1e-6; tests.append('FPS and radius-neighbour constraints')
    m=PointNet2(centroids=(32,16,8,4)).eval(); x=torch.randn(2,4,128); perm=torch.randperm(128)
    with torch.no_grad(): out=m(x); permuted=m(x[:,:,perm]); assert out.shape==(2,19,128); assert torch.allclose(out[:,:,perm],permuted,atol=5e-5)
    m.train(); loss=m(x).square().mean(); loss.backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None); tests.append('Hierarchical SA/FP output shape, permutation equivariance, finite gradients')
    cm=np.zeros((19,19),np.int64); cm[0,0]=2; cm[0,1]=1; cm[1,1]=1; r=scores(cm); assert np.isclose(r['miou19'],(2/3+.5)/19); assert np.isclose(r['accuracy'],.75); tests.append('19-class averaging and hand-computed confusion arithmetic')
    mp=yaml.safe_load((ROOT/'semantic-kitti.yaml').read_text())['learning_map']; assert mp[10]==mp[252]==1 and mp[0]==mp[1]==mp[99]==0 and mp[40]==mp[60]==9; tests.append('Official moving-class merges and ignored labels')
    class Constant(torch.nn.Module):
        def forward(self,x):
            z=torch.zeros((len(x),19,x.shape[-1]),device=x.device); z[:,8]=1; return z
    points=np.random.default_rng(1).normal(size=(47,4)).astype(np.float32); points[:,:2]*=20; stats={'remission_mean':.2,'remission_std':.1}
    p,c,t=predict_scene(Constant().cuda(),points,stats,points=64,batch=4); assert len(p)==47 and (p==8).all() and t['covered_points']==47 and t['votes_per_point']==2; tests.append('Sparse-cell padding and exact twice-voted full-scene coverage')
    over=json.loads((ROOT/'overfit.json').read_text()); assert over['accuracy']>.95 and over['final_loss']<over['initial_loss']/10; tests.append('Actual multi-class overfit sanity check')
    result={'passed':True,'tests':tests}; (ROOT/'verification.json').write_text(json.dumps(result,indent=2)); print(result)
if __name__=='__main__': main()
