import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time,random,sys,platform,subprocess
from pathlib import Path
import numpy as np,torch,psutil
from torch.utils.data import DataLoader
from model import PointNet2
from data import ROOT,Blocks,class_weights

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True,warn_only=True)

def scores(cm):
    cm=np.asarray(cm,dtype=np.int64); tp=np.diag(cm).astype(float); gt=cm.sum(1); pred=cm.sum(0); union=gt+pred-tp
    def div(a,b): return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    iou=div(tp,union)
    return {'miou19':float(iou.mean()),'miou_present_union':float(iou[union>0].mean()) if (union>0).any() else 0.,'accuracy':float(tp.sum()/max(gt.sum(),1)),'iou':iou.tolist(),'precision':div(tp,pred).tolist(),'recall':div(tp,gt).tolist(),'f1':div(2*tp,gt+pred).tolist(),'support':gt.tolist(),'predicted_support':pred.tolist(),'confusion':cm.tolist(),'zero_union':np.flatnonzero(union==0).tolist()}

@torch.no_grad()
def evaluate(m,loader):
    m.eval(); cm=np.zeros((19,19),np.int64); losses=[]
    for x,y in loader:
        x=x.cuda(); y=y.cuda()
        with torch.autocast('cuda',dtype=torch.float16): logits=m(x)
        p=logits.argmax(1); v=y>=0; cm+=torch.bincount(19*y[v]+p[v],minlength=361).view(19,19).cpu().numpy()
    return scores(cm)

def state_dict(m,opt,scheduler,scaler,cfg,epoch,step,best,history,seconds):
    return {'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'config':cfg,'epoch':epoch,'step':step,'best_val_miou19':best,'history':history,'training_wall_seconds':seconds,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}

def train(cfg,resume=None,warmstart=None):
    seed_all(cfg['seed']); run=ROOT/'runs'/cfg['name']; run.mkdir(parents=True,exist_ok=True); (run/'config.json').write_text(json.dumps(cfg,indent=2))
    m=PointNet2().cuda(); opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['min_lr']); scaler=torch.amp.GradScaler('cuda')
    dataset=cfg.get('dataset','train'); trainset=Blocks(dataset,cfg['points'],cfg['seed'],cfg.get('train_limit'),True,stats_name=dataset); valset=Blocks('val',cfg['points'],123,cfg['val_blocks'],False,stats_name=dataset)
    (run/'train_stats.json').write_text(json.dumps(trainset.stats,indent=2))
    weights=torch.from_numpy(class_weights(dataset)).cuda(); criterion=torch.nn.CrossEntropyLoss(weight=weights,ignore_index=-1,label_smoothing=cfg.get('label_smoothing',0.))
    history=[]; best=-1.; start_epoch=0; totalsteps=0; previous_seconds=0.
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=True); m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler']); history=ck['history']; best=ck['best_val_miou19']; start_epoch=ck['epoch']; totalsteps=ck['step']; previous_seconds=ck['training_wall_seconds']; torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif warmstart:
        ck=torch.load(warmstart,map_location='cpu',weights_only=True); m.load_state_dict(ck['model']); (run/'warmstart.json').write_text(json.dumps({'checkpoint':str(Path(warmstart).resolve()),'epoch':ck['epoch'],'validation_miou19':ck['best_val_miou19']},indent=2))
    val_loader=DataLoader(valset,batch_size=cfg['batch'],num_workers=0); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    (run/'class_weights.json').write_text(json.dumps(weights.cpu().tolist()))
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':str(torch.cuda.get_device_properties(0)),'ram':dict(psutil.virtual_memory()._asdict()),'command':subprocess.list2cmdline(sys.argv),'parameters':sum(p.numel() for p in m.parameters()),'train_blocks':len(trainset),'validation_blocks':len(valset)}; (run/'environment.json').write_text(json.dumps(env,indent=2))
    try:
        for epoch in range(start_epoch,cfg['epochs']):
            trainset.epoch=epoch; gen=torch.Generator().manual_seed(cfg['seed']+epoch); loader=DataLoader(trainset,batch_size=cfg['batch'],shuffle=True,generator=gen,num_workers=0,drop_last=False); m.train(); losses=[]; epochstart=time.perf_counter(); lastlog=time.perf_counter(); nsteps=0
            for step,(x,y) in enumerate(loader):
                if cfg.get('max_steps_per_epoch') and step>=cfg['max_steps_per_epoch']: break
                if not (y>=0).any(): continue
                x=x.cuda(); y=y.cuda(); opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16): loss=criterion(m(x),y)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),10.); scaler.step(opt); scaler.update(); losses.append(float(loss)); totalsteps+=1; nsteps+=1
                if time.perf_counter()-lastlog>30:
                    progress={'epoch':epoch+1,'step_in_epoch':step+1,'total_steps':totalsteps,'loss_recent':float(np.mean(losses[-50:])),'elapsed_seconds':previous_seconds+time.perf_counter()-start,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20}
                    (run/'progress.json').write_text(json.dumps(progress,indent=2)); print(progress,flush=True); lastlog=time.perf_counter()
            torch.cuda.synchronize(); trainsec=time.perf_counter()-epochstart; r=evaluate(m,val_loader); scheduler.step()
            row={'epoch':epoch+1,'steps':nsteps,'total_steps':totalsteps,'loss':float(np.mean(losses)),'val_miou19':r['miou19'],'val_accuracy':r['accuracy'],'val_car_iou':r['iou'][0],'val_road_iou':r['iou'][8],'train_seconds':trainsec,'lr_next':scheduler.get_last_lr()[0],'wall_seconds':previous_seconds+time.perf_counter()-start,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20}; history.append(row)
            (run/'history.json').write_text(json.dumps(history,indent=2)); print(row,flush=True)
            if r['miou19']>best:
                best=r['miou19']; ck=state_dict(m,opt,scheduler,scaler,cfg,epoch+1,totalsteps,best,history,previous_seconds+time.perf_counter()-start); torch.save(ck,run/'best.pt'); (run/'best_validation.json').write_text(json.dumps(r,indent=2))
            ck=state_dict(m,opt,scheduler,scaler,cfg,epoch+1,totalsteps,best,history,previous_seconds+time.perf_counter()-start); torch.save(ck,run/'last.tmp'); os.replace(run/'last.tmp',run/'last.pt')
            best_epoch=max(history,key=lambda h:h['val_miou19'])['epoch']
            if epoch+1>=cfg.get('minimum_epochs',cfg['epochs']) and epoch+1-best_epoch>=cfg.get('early_stop_patience',cfg['epochs']):
                print('Validation-only early stopping',flush=True); break
        (run/'complete.json').write_text(json.dumps({'epochs':history[-1]['epoch'],'planned_maximum_epochs':cfg['epochs'],'updates':totalsteps,'wall_seconds':previous_seconds+time.perf_counter()-start,'best_validation_miou19':best,'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20},indent=2))
    except BaseException as e:
        (run/'failure.json').write_text(json.dumps({'error':repr(e),'time':time.time(),'last_completed_epoch':history[-1]['epoch'] if history else 0},indent=2)); raise

def overfit():
    seed_all(19); ds=Blocks('train',4096,19,augment=False); chosen=[]
    for i,b in enumerate(ds.blocks):
        x,y=ds[i]
        if len(torch.unique(y[y>=0]))>=5 and (y==0).sum()>50: chosen.append((x,y,b))
        if len(chosen)==2: break
    assert len(chosen)==2
    x=torch.stack([a[0] for a in chosen]).cuda(); y=torch.stack([a[1] for a in chosen]).cuda(); m=PointNet2().cuda(); opt=torch.optim.AdamW(m.parameters(),lr=.003); scaler=torch.amp.GradScaler('cuda'); lossfn=torch.nn.CrossEntropyLoss(ignore_index=-1); losses=[]; t=time.perf_counter()
    for step in range(240):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss=lossfn(m(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss))
        if step%40==0: print('overfit',step,losses[-1],flush=True)
    m.eval()
    with torch.no_grad(): p=m(x).argmax(1); v=y>=0; acc=float((p[v]==y[v]).float().mean())
    out={'initial_loss':losses[0],'final_loss':losses[-1],'accuracy':acc,'seconds':time.perf_counter()-t,'steps':len(losses),'blocks':[a[2] for a in chosen],'classes':torch.unique(y[y>=0]).cpu().tolist(),'losses':losses}; (ROOT/'overfit.json').write_text(json.dumps(out,indent=2)); torch.save({'model':m.state_dict(),'purpose':'overfit diagnostic only'},ROOT/'overfit.pt'); print({k:v for k,v in out.items() if k!='losses'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--resume'); p.add_argument('--warmstart'); p.add_argument('--overfit',action='store_true'); a=p.parse_args()
    if a.overfit: overfit()
    else: train(json.loads(Path(a.config).read_text()),a.resume,a.warmstart)
