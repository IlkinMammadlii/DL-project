"""Cache complete XYZ/remission/targets, and label-independent local block indices.
All sequence-07 scans train; all validation/test scans remain separately stored.
"""
from pathlib import Path
import sys,json,tarfile,time,shutil,argparse
import numpy as np,yaml
ROOT=Path(__file__).resolve().parent
from dataio import safe_load

def main(extra06=False):
    cache=ROOT/'cache'; cache.mkdir(exist_ok=True)
    mappingfile=ROOT/'semantic-kitti.yaml'
    if not mappingfile.exists(): shutil.copyfile(ROOT.parent/'roadseg/semantic-kitti.yaml',mappingfile)
    mapping=yaml.safe_load(mappingfile.read_text()); lut=np.full(65536,-1,np.int16)
    for k,v in mapping['learning_map'].items(): lut[k]=v-1
    if not (ROOT/'SEMANTICKITTI-API-LICENSE.txt').exists(): shutil.copyfile(ROOT.parent/'roadseg/API-LICENSE.txt',ROOT/'SEMANTICKITTI-API-LICENSE.txt')
    names=[mapping['labels'][mapping['learning_map_inv'][i]] for i in range(1,20)]; (ROOT/'classes.json').write_text(json.dumps(names))
    for seq,split in ([('06','train06')] if extra06 else [('07','train'),('04','val'),('03','test')]):
        dest=cache/seq; dest.mkdir(exist_ok=True); blocks=[]; scans=[]; counts=np.zeros(19,np.int64); rsum=0.; r2sum=0.; n=0; ignored=0; t0=time.perf_counter()
        archive=ROOT/'archives'/f'{seq}.tar.gz'
        if not archive.exists(): archive=ROOT.parent/f'roadseg/data/{seq}.tar.gz'
        with tarfile.open(archive,'r|gz') as tar:
            for mem in tar:
                if not mem.name.endswith('.pth'): continue
                scan=int(Path(mem.name).stem); d=safe_load(tar.extractfile(mem)); x=np.column_stack((d['coord'],np.asarray(d['remission']).reshape(-1))).astype(np.float32); raw=np.asarray(d['semantic_label_raw']).reshape(-1).astype(np.int64)&65535; y=lut[raw]
                assert len(x)==len(y) and np.isfinite(x).all(),mem.name
                counts+=np.bincount(y[y>=0],minlength=19); ignored+=int((y<0).sum()); rsum+=float(x[:,3].astype(float).sum()); r2sum+=float(np.square(x[:,3].astype(float)).sum()); n+=len(x)
                np.save(dest/f'{scan:06d}_x.npy',x); np.save(dest/f'{scan:06d}_y.npy',y)
                scans.append({'sequence':seq,'scan':scan,'points':len(x),'member':mem.name})
                # Train/validation spatial cells: shifted grids add boundary variety.
                # Test block selection is deferred to full-scene inference, never labels.
                if split!='test':
                    for shift in ([0.,8.] if split.startswith('train') else [0.]):
                        cell=np.floor((x[:,:2]-shift)/16).astype(np.int32); cells,inv=np.unique(cell,axis=0,return_inverse=True); order=np.argsort(inv,kind='stable').astype(np.int32); cuts=np.r_[0,np.flatnonzero(np.diff(inv[order]))+1,len(order)]
                        np.save(dest/f'{scan:06d}_order{int(shift)}.npy',order)
                        for j,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
                            if b-a<128: continue
                            blocks.append({'sequence':seq,'scan':scan,'shift':int(shift),'start':int(a),'stop':int(b),'cx':float(cells[j,0]*16+8+shift),'cy':float(cells[j,1]*16+8+shift)})
                if len(scans)%100==0: print(split,len(scans),'scans',len(blocks),'blocks',flush=True)
        scans.sort(key=lambda q:q['scan']); blocks.sort(key=lambda q:(q['scan'],q['shift'],q['start']))
        (ROOT/f'{split}_scans.json').write_text(json.dumps(scans)); (ROOT/f'{split}_blocks.json').write_text(json.dumps(blocks))
        stats={'sequence':seq,'scans':len(scans),'blocks':len(blocks),'counts':counts.tolist(),'ignored':ignored,'points':n,'remission_mean':rsum/n,'remission_std':float(np.sqrt(r2sum/n-(rsum/n)**2)),'seconds':time.perf_counter()-t0}
        (ROOT/f'{split}_stats.json').write_text(json.dumps(stats,indent=2)); print(stats,flush=True)
    if extra06:
        a=json.loads((ROOT/'train_stats.json').read_text()); b=json.loads((ROOT/'train06_stats.json').read_text()); n=a['points']+b['points']; mean=(a['remission_mean']*a['points']+b['remission_mean']*b['points'])/n; second=sum((s['remission_std']**2+s['remission_mean']**2)*s['points'] for s in [a,b])/n
        joint={'sequences':['06','07'],'scans':a['scans']+b['scans'],'blocks':a['blocks']+b['blocks'],'counts':(np.array(a['counts'])+np.array(b['counts'])).tolist(),'points':n,'ignored':a['ignored']+b['ignored'],'remission_mean':mean,'remission_std':float(np.sqrt(second-mean**2))}
        for kind in ['blocks','scans']: (ROOT/f'train_joint_{kind}.json').write_text(json.dumps(json.loads((ROOT/f'train_{kind}.json').read_text())+json.loads((ROOT/f'train06_{kind}.json').read_text())))
        (ROOT/'train_joint_stats.json').write_text(json.dumps(joint,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--extra06',action='store_true'); main(p.parse_args().extra06)
