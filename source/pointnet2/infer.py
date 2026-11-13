"""Complete-scene tiling and forward inference. Every point receives direct logits."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,torch
from model import PointNet2
from data import ROOT
from train import scores,seed_all

def load(checkpoint):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True); m=PointNet2().cuda().eval(); m.load_state_dict(ck['model']); return m,ck

@torch.inference_mode()
def predict_scene(m,x,stats,scan=0,points=4096,batch=4,shifts=(0.,8.)):
    start=time.perf_counter(); sums=np.zeros((len(x),19),np.float32); votes=np.zeros(len(x),np.uint8); forward_seconds=0.; batches=0; tasks=[]
    def execute():
        nonlocal forward_seconds,batches
        if not tasks: return
        inp=torch.from_numpy(np.stack([t[0] for t in tasks])).cuda()
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16): logits=m(inp).float().softmax(1)
        torch.cuda.synchronize(); forward_seconds+=time.perf_counter()-t; prob=logits.transpose(1,2).cpu().numpy()
        for j,(_,ids,n) in enumerate(tasks): sums[ids[:n]]+=prob[j,:n]; votes[ids[:n]]+=1
        batches+=1; tasks.clear()
    rng=np.random.default_rng(17000+scan)
    for shift in shifts:
        cell=np.floor((x[:,:2]-shift)/16).astype(np.int32); cells,inv=np.unique(cell,axis=0,return_inverse=True); order=np.argsort(inv,kind='stable'); cuts=np.r_[0,np.flatnonzero(np.diff(inv[order]))+1,len(order)]
        for j,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
            pool=order[a:b]; ids=rng.permutation(pool)
            for k in range(0,len(ids),points):
                query=ids[k:k+points]; n=len(query)
                if n<points: query=np.r_[query,rng.choice(pool,points-n,replace=True)]
                feat=np.array(x[query],copy=True); centre=cells[j]*16+8+shift; feat[:,:2]-=centre; feat[:,3]=(feat[:,3]-stats['remission_mean'])/stats['remission_std']; tasks.append((feat.T.copy(),query,n))
                if len(tasks)==batch: execute()
    execute(); assert (votes==len(shifts)).all(), 'Scene coverage failed'
    prob=sums/votes[:,None]; pred=prob.argmax(1).astype(np.uint8); confidence=prob.max(1).astype(np.float16)
    return pred,confidence,{'points':len(x),'covered_points':int((votes>0).sum()),'votes_per_point':len(shifts),'forward_seconds':forward_seconds,'wall_seconds':time.perf_counter()-start,'batches':batches}

def scene_metric(y,p):
    valid=y>=0; return scores(np.bincount(19*y[valid].astype(np.int64)+p[valid],minlength=361).reshape(19,19))

def evaluate(checkpoint,split='test',outname='final',scans=None):
    seed_all(42); m,ck=load(checkpoint); dest=ROOT/'evaluation'/outname/split; dest.mkdir(parents=True,exist_ok=True); seq='03' if split=='test' else '04'; statfile=Path(checkpoint).parent/'train_stats.json'; stats=json.loads((statfile if statfile.exists() else ROOT/'train_stats.json').read_text()); items=json.loads((ROOT/f'{split}_scans.json').read_text()); cm=np.zeros((19,19),np.int64); far_cm=np.zeros_like(cm); timings=[]
    if scans is not None: items=[q for q in items if q['scan'] in scans]
    marker=dest/'checkpoint.json'; identity={'path':str(Path(checkpoint).resolve()),'sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'epoch':ck.get('epoch'),'validation_miou19':ck.get('best_val_miou19'),'scan_ids':[q['scan'] for q in items],'split':split,'selection':'All sequence scans' if scans is None else 'Explicit numeric scan list; independent of labels'}
    if marker.exists(): assert json.loads(marker.read_text())==identity,'Checkpoint/selection changed: choose a new output name'
    else: marker.write_text(json.dumps(identity,indent=2))
    for item in items:
        scan=item['scan']; x=np.load(ROOT/'cache'/seq/f'{scan:06d}_x.npy',mmap_mode='r'); y=np.load(ROOT/'cache'/seq/f'{scan:06d}_y.npy',mmap_mode='r'); destfile=dest/f'{scan:06d}.npz'; timingfile=dest/f'{scan:06d}.json'
        if destfile.exists() and timingfile.exists():
            with np.load(destfile) as d: p=d['prediction']
            timing=json.loads(timingfile.read_text())
        else:
            p,c,timing=predict_scene(m,x,stats,scan); np.savez_compressed(destfile,prediction=p,confidence=c); r=scene_metric(y,p); timing.update(sequence=seq,scan=scan,miou19=r['miou19'],accuracy=r['accuracy'],car_iou=r['iou'][0],car_support=r['support'][0]); timingfile.write_text(json.dumps(timing,indent=2))
        valid=y>=0; cm+=np.bincount(19*y[valid].astype(np.int64)+p[valid],minlength=361).reshape(19,19)
        far=valid&(np.linalg.norm(x[:,:3],axis=1)>30); far_cm+=np.bincount(19*y[far].astype(np.int64)+p[far],minlength=361).reshape(19,19); timings.append(timing)
        print(split,scan,timing,flush=True)
        result=scores(cm); result.update(scans_completed=len(timings),scans_requested=len(items),forward_seconds=sum(t['forward_seconds'] for t in timings),inference_wall_seconds=sum(t['wall_seconds'] for t in timings),total_points=sum(t['points'] for t in timings),farther_than_30m=scores(far_cm),per_scene_timings=timings); (dest/'metrics.json').write_text(json.dumps(result,indent=2))
    (dest/'complete.json').write_text(json.dumps({'scans':len(items),'checkpoint_sha256':identity['sha256']},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/best.pt')); p.add_argument('--split',choices=['val','test'],default='test'); p.add_argument('--outname',default='final'); p.add_argument('--scans',type=int,nargs='*'); a=p.parse_args(); evaluate(a.checkpoint,a.split,a.outname,a.scans)
