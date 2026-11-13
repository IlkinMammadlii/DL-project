"""Matched camera, sparse LiDAR, range, semantic and error views; recorded video.
Pillow rasterisation keeps full-scene/video rendering lightweight on low RAM.
"""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from data import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG=(15,25,34); INK=(236,241,244); MUTED=(180,194,202)
PALETTE=np.array([[0,140,220],[230,159,0],[204,121,167],[213,94,0],[86,180,233],[240,200,70],[210,110,180],[170,120,200],[145,155,160],[150,110,70],[204,121,167],[180,160,120],[230,160,65],[130,150,160],[0,175,115],[175,140,95],[160,175,80],[230,100,100],[95,210,210]],dtype=np.uint8)
def font(size):
    for p in ['C:/Windows/Fonts/segoeui.ttf','C:/Windows/Fonts/arial.ttf']:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default(size=size)
def semantic(y):
    out=np.tile([70,80,88],(len(y),1)).astype(np.uint8); valid=y>=0; out[valid]=PALETTE[y[valid]]; return out
def intensity(x):
    rem=np.clip(x[:,3]/.8,0,1); return (plt.get_cmap('cividis')(rem)[:,:3]*255).astype(np.uint8)
def calib(seq='03'):
    p=ROOT/'camera'/f'{seq}_calib.txt'
    if not p.exists(): return None
    lines={k:np.fromstring(v,sep=' ') for k,v in [l.split(':',1) for l in p.read_text().splitlines() if ':' in l]}
    tr=np.eye(4); tr[:3]=lines['Tr'].reshape(3,4); return lines['P2'].reshape(3,4)@tr
def image_projection(x,size=(1242,375),seq='03'):
    P=calib(seq)
    if P is not None:
        q=np.column_stack([x[:,:3],np.ones(len(x))])@P.T; depth=q[:,2]; u=q[:,0]/np.maximum(depth,1e-8); v=q[:,1]/np.maximum(depth,1e-8)
    else:
        depth=x[:,0]; u=size[0]/2-700*x[:,1]/np.maximum(depth,1e-8); v=size[1]/2-700*x[:,2]/np.maximum(depth,1e-8)
    valid=(depth>.2)&(u>=0)&(v>=0)&(u<size[0])&(v<size[1]); ids=np.flatnonzero(valid); return np.column_stack([u[ids],v[ids]]),ids,depth[ids]
def raster(coords,ids,rgb,width,height,depth=None,background=None,radius=0):
    canvas=np.zeros((height,width,3),np.uint8); canvas[:]=BG
    if background is not None: canvas=np.asarray(background.resize((width,height))).copy()
    if len(ids)==0: return Image.fromarray(canvas)
    uv=np.floor(coords).astype(int); valid=(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<width)&(uv[:,1]<height); uv=uv[valid]; ids=ids[valid]
    if depth is not None:
        dep=np.asarray(depth)[valid]; order=np.argsort(dep)[::-1]; uv=uv[order]; ids=ids[order]
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            xx=uv[:,0]+dx; yy=uv[:,1]+dy; ok=(xx>=0)&(xx<width)&(yy>=0)&(yy<height); canvas[yy[ok],xx[ok]]=rgb[ids[ok]]
    return Image.fromarray(canvas)
def driver(x,rgb,size=(1242,375),seq='03',background=None):
    uv,ids,d=image_projection(x,size,seq); return raster(uv,ids,rgb,*size,depth=d,background=background,radius=1)
def topdown(x,rgb,width=620,height=500):
    # Forward is up, driver-right is right; fixed geometry across all scans.
    u=(-x[:,1]+35)/70*(width-1); v=(75-x[:,0])/100*(height-1); uv=np.column_stack([u,v]); im=raster(uv,np.arange(len(x)),rgb,width,height,radius=0)
    draw=ImageDraw.Draw(im)
    for forward in [0,20,40,60]:
        yy=int((75-forward)/100*(height-1)); draw.line((0,yy,width,yy),fill=(45,55,64)); draw.text((5,yy+3),f'{forward} m forward',font=font(12),fill=MUTED)
    ego=(int(width*.5),int(height*.75)); draw.polygon([(ego[0],ego[1]-10),(ego[0]-6,ego[1]+6),(ego[0]+6,ego[1]+6)],fill='white'); return im
def range_image(x,rgb,width=1024,height=64):
    distance=np.linalg.norm(x[:,:3],axis=1); yaw=np.arctan2(x[:,1],x[:,0]); pitch=np.arcsin(x[:,2]/np.maximum(distance,1e-8)); u=(1-yaw/np.pi)*.5*width; v=(np.deg2rad(3)-pitch)/np.deg2rad(28)*height; uv=np.column_stack([u,v]); return raster(uv,np.arange(len(x)),rgb,width,height,depth=distance).resize((width,256),Image.Resampling.NEAREST)
def panel(im,title,caption,size=(640,390)):
    out=Image.new('RGB',size,BG); d=ImageDraw.Draw(out); d.text((16,10),title,font=font(21),fill=INK)
    avail=(size[0]-24,size[1]-90); im=im.copy(); im.thumbnail(avail,Image.Resampling.LANCZOS); out.paste(im,((size[0]-im.width)//2,46+(avail[1]-im.height)//2)); d.text((14,size[1]-32),caption,font=font(14),fill=MUTED); return out
def legend(canvas,y0):
    names=json.loads((ROOT/'classes.json').read_text()); d=ImageDraw.Draw(canvas); chosen=[8,10,0,12,14,13,16,17,18]
    x=20
    for i in chosen:
        name=names[i]; d.rectangle((x,y0,x+14,y0+14),fill=tuple(PALETTE[i])); d.text((x+21,y0-3),name,font=font(17),fill=INK); x+=int(d.textlength(name,font=font(17)))+48
    d.text((20,y0+30),'Other SemanticKITTI classes retain their own colours. Greyed-out truth points are ignored labels.',font=font(15),fill=MUTED)
def error_colors(y,p):
    out=np.tile([75,85,93],(len(y),1)).astype(np.uint8); valid=y>=0; out[valid&(p==y)]=[160,190,205]; out[valid&(p!=y)]=[255,100,45]; return out
def legend_picture():
    names=json.loads((ROOT/'classes.json').read_text()); im=Image.new('RGB',(1100,540),BG); d=ImageDraw.Draw(im); d.text((22,14),'Semantic colours: human labels and model predictions',font=font(27),fill=INK); records=[]
    for i,name in enumerate(names):
        xx=25+(i//10)*550; yy=65+(i%10)*43; d.rectangle((xx,yy,xx+22,yy+22),fill=tuple(PALETTE[i])); d.text((xx+35,yy-4),f'{i+1}  {name}',font=font(23),fill=INK); records.append({'model_id':i,'official_learning_id':i+1,'class':name,'rgb':PALETTE[i].tolist()})
    im.save(ROOT/'figures/class_legend.png'); (ROOT/'class_palette.json').write_text(json.dumps(records,indent=2))
def sensor_check(scan=100):
    camfile=ROOT/'camera/03'/f'{scan:06d}.png'
    if not camfile.exists(): return
    x=np.load(ROOT/'cache/03'/f'{scan:06d}_x.npy'); y=np.load(ROOT/'cache/03'/f'{scan:06d}_y.npy'); cam=Image.open(camfile).convert('RGB'); im=Image.new('RGB',(1920,480),BG)
    tiles=[(cam,'Real camera image',f'Sequence 03, scan {scan:06d}'),(driver(x,intensity(x),cam.size),'Sparse LiDAR intensity projection','Blue = weak / yellow = strong return; not a photograph'),(driver(x,semantic(y),cam.size,background=cam),'Calibration check: ground-truth labels','Projected human labels, not model predictions')]
    for i,(a,b,c) in enumerate(tiles): im.paste(panel(a,b,c),(i*640,40))
    ImageDraw.Draw(im).text((18,8),'The camera provides context. PointNet++ receives XYZ and remission only.',font=font(24),fill=INK); im.save(ROOT/'figures/sensor_alignment.png')

def scene_views(scan,split='test',outname='final'):
    seq='03' if split=='test' else '04'; x=np.load(ROOT/'cache'/seq/f'{scan:06d}_x.npy'); y=np.load(ROOT/'cache'/seq/f'{scan:06d}_y.npy'); data=np.load(ROOT/'evaluation'/outname/split/f'{scan:06d}.npz'); p=data['prediction']; error=error_colors(y,p); camfile=ROOT/'camera'/seq/f'{scan:06d}.png'; camera=Image.open(camfile).convert('RGB') if camfile.exists() else None; size=camera.size if camera else (1242,375)
    cam=camera if camera else Image.new('RGB',size,BG)
    if camera is None: ImageDraw.Draw(cam).text((30,100),'Matching RGB camera frame unavailable',font=font(26),fill=INK)
    return x,y,p,cam,size,error,camera is not None

def static(scan,split='test',outname='final'):
    seq='03' if split=='test' else '04'; x,y,p,cam,size,err,hascam=scene_views(scan,split,outname); truth=semantic(y); pred=semantic(p); sensor=intensity(x)
    tiles=[(cam,'1  Scene context','Real KITTI RGB frame' if hascam else 'No synthetic camera image is substituted'),(driver(x,sensor,size,seq),'2  What the sensor measures','Intensity: blue = weak, yellow = strong; not a photo'),(range_image(x,sensor),'3  LiDAR range-image layout','360-degree angular grid; blue = weak, yellow = strong'),(driver(x,truth,size,seq),'4  Ground truth: driver-facing','Human semantic labels on projected LiDAR points'),(driver(x,pred,size,seq),'5  PointNet++ prediction: driver-facing','Per-point categories from XYZ and remission'),(driver(x,err,size,seq),'6  Errors: driver-facing','Orange = wrong; pale blue = correct; dark = ignored'),(topdown(x,truth),'7  Ground truth: top-down','Forward is up; white triangle marks the sensor'),(topdown(x,pred),'8  Prediction: top-down','Same points and viewpoint as ground truth'),(topdown(x,err),'9  Errors: top-down','All scan points inferred; view displays a fixed crop')]
    canvas=Image.new('RGB',(1920,1300),BG); d=ImageDraw.Draw(canvas)
    title=f'Understanding Driving Scenes with PointNet++  |  sequence {seq}, scan {scan:06d}'
    if outname!='final':
        meta=json.loads((ROOT/'evaluation'/outname/split/'checkpoint.json').read_text()); title=f"PointNet++ EARLY training preview: epoch {meta['epoch']}  |  sequence {seq}, scan {scan:06d}"
    d.text((20,12),title,font=font(27),fill=INK)
    for i,(im,title,cap) in enumerate(tiles): canvas.paste(panel(im,title,cap),(i%3*640,60+i//3*390))
    legend(canvas,1240); dest=ROOT/'figures'/outname; dest.mkdir(parents=True,exist_ok=True); canvas.save(dest/f'{seq}_{scan:06d}_scene.png')
    if hascam:
        over=driver(x,pred,size,seq,background=cam); comparison=Image.new('RGB',(size[0]*2,size[1]+70),BG); comparison.paste(cam,(0,50)); comparison.paste(over,(size[0],50)); d=ImageDraw.Draw(comparison); d.text((15,12),'Real camera image',font=font(24),fill=INK); d.text((size[0]+15,12),'Predicted LiDAR labels over RGB: projected points only',font=font(24),fill=INK); comparison.save(dest/f'{seq}_{scan:06d}_camera_overlay.png')
    # Binary PLY with both target/prediction and RGB, usable in common cloud viewers.
    if scan in [0,200,400,600,800]:
        arr=np.empty(len(x),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('prediction','u1'),('truth','<i2')]); arr['x'],arr['y'],arr['z']=x[:,:3].T; arr['red'],arr['green'],arr['blue']=pred.T; arr['prediction']=p; arr['truth']=y
        with open(dest/f'{seq}_{scan:06d}.ply','wb') as f:
            f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(x)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar prediction\nproperty short truth\nend_header\n'.encode()); f.write(arr.tobytes())
    print('static',scan,flush=True)

def vehicle_details(outname='final'):
    """Post-hoc qualitative regions; selection uses labels only for explanation."""
    candidates=[]
    for scan in [0,100,130,160,200,230,260,299,400,600,800]:
        x,y,p,cam,size,err,hascam=scene_views(scan,'test',outname)
        if not hascam: continue
        uv,visible,_=image_projection(x,size); dist=np.linalg.norm(x[:,:3],axis=1); ids=visible[(y[visible]==0)&(dist[visible]>30)&(dist[visible]<65)]
        if not len(ids): continue
        sample=ids[::max(1,len(ids)//20)]; counts=[int(((np.linalg.norm(x[:,:3]-x[k,:3],axis=1)<3)&(y==0)).sum()) for k in sample]; anchor=int(sample[np.argmax(counts)]); region=(np.linalg.norm(x[:,:3]-x[anchor,:3],axis=1)<3)&(y==0)
        if region.sum()<5: continue
        pos=uv[np.flatnonzero(visible==anchor)[0]]; box=(max(0,int(pos[0]-110)),max(0,int(pos[1]-80)),min(size[0],int(pos[0]+110)),min(size[1],int(pos[1]+80))); recall=float((p[region]==0).mean()); candidates.append({'scan':scan,'anchor':anchor,'distance_m':float(dist[anchor]),'car_points_in_region':int(region.sum()),'car_recall':recall,'box':box})
    candidates.sort(key=lambda q:q['car_recall']); chosen=[candidates[i] for i in sorted(set([0,len(candidates)//2,len(candidates)-1]))] if candidates else []
    dest=ROOT/'figures'/outname; records=[]
    for q in chosen:
        scan=q['scan']; x,y,p,cam,size,err,_=scene_views(scan,'test',outname); box=tuple(q['box']); canvas=Image.new('RGB',(1280,790),BG); d=ImageDraw.Draw(canvas); d.text((18,10),f"Distant vehicle region: scan {scan:06d}, {q['distance_m']:.1f} m from sensor",font=font(26),fill=INK)
        panels=[(cam.crop(box),'Real camera crop','Enlarged photograph of the selected vehicle region'),(driver(x,semantic(y),size).crop(box),'Human ground truth','Human car labels locate this diagnostic region'),(driver(x,semantic(p),size).crop(box),'PointNet++ prediction',f"Car-point recall in 3 m region: {q['car_recall']:.1%}"),(driver(x,err,size).crop(box),'Errors at the same viewpoint','Orange = wrong; pale blue = correct')]
        for i,(im,title,cap) in enumerate(panels): canvas.paste(panel(im,title,cap,(640,340)),(i%2*640,55+i//2*340))
        d.text((18,748),'Post-hoc qualitative selection using ground truth; region is not an instance-level benchmark.',font=font(18),fill=MUTED); canvas.save(dest/f'03_{scan:06d}_distant_vehicle.png'); records.append(q)
    (dest/'distant_vehicle_regions.json').write_text(json.dumps({'selection':'Worst, median and best car recall among available camera scenes with labelled car points 30–65 m away; 3 m diagnostic region, not object instances','regions':records},indent=2))

def video(outname='final',start=100,stop=300):
    dest=ROOT/'video'/outname; dest.mkdir(parents=True,exist_ok=True); frames=ROOT.parent.parent/'work/pointnet2_video_frames'/outname; frames.mkdir(parents=True,exist_ok=True); timings=[]
    for scan in range(start,stop):
        x,y,p,cam,size,err,hascam=scene_views(scan,'test',outname); sensor=intensity(x); pred=semantic(p); frame=Image.new('RGB',(1920,1080),BG); d=ImageDraw.Draw(frame)
        clock=scan*.1; ts=ROOT/'camera/03_times.txt'
        if ts.exists(): clock=float(np.loadtxt(ts)[scan])
        d.text((20,12),f'PointNet++ recorded-sequence demonstration  |  sequence 03  |  scan {scan:06d}  |  sensor time {clock:.2f} s',font=font(26),fill=INK)
        tiles=[(cam,'Synchronized real camera','RGB frame from the same sequence and scan ID' if hascam else 'Matching camera unavailable'),(driver(x,sensor,size),'LiDAR input: driver-facing','Sparse intensity: blue = weak, yellow = strong'),(driver(x,pred,size),'Semantic prediction: driver-facing','PointNet++ per-point classes'),(driver(x,pred,size,background=cam) if hascam else range_image(x,sensor),'Projected labels over RGB' if hascam else '360-degree range image','Only projected LiDAR points receive labels'),(topdown(x,pred,width=600,height=430),'Semantic prediction: top-down','Forward is up; white triangle is the sensor'),(driver(x,err,size),'Prediction errors','Orange = wrong; pale blue = correct')]
        for i,(im,title,cap) in enumerate(tiles): frame.paste(panel(im,title,cap,size=(640,450)),(i%3*640,65+i//3*450))
        legend(frame,975); d.text((20,1040),'Playback: 10 sensor frames/s. This recorded video does not imply real-time inference.',font=font(19),fill=INK); frame.save(frames/f'{scan-start:06d}.png')
        timing=json.loads((ROOT/'evaluation'/outname/'test'/f'{scan:06d}.json').read_text()); timings.append(timing)
    output=dest/'driving_scene_demo.mp4'; subprocess.run(['ffmpeg','-y','-framerate','10','-i',str(frames/'%06d.png'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True,stdout=subprocess.DEVNULL,stderr=open(dest/'ffmpeg.log','w'))
    secs=[r['wall_seconds'] for r in timings]; report={'frames':len(timings),'first_scan':start,'last_scan':stop-1,'playback_fps':10,'playback_seconds':len(timings)/10,'mean_processing_seconds_per_scan':float(np.mean(secs)),'median_processing_seconds_per_scan':float(np.median(secs)),'p95_processing_seconds_per_scan':float(np.percentile(secs,95)),'processing_fps':1/float(np.mean(secs)),'real_time_at_10hz':bool(np.percentile(secs,95)<=.1),'timing_excludes':'video rendering and camera reads; includes block preparation, transfers, GPU forward, voting','camera_alignment':'Exact KITTI sequence-03 image_2 scan IDs with odometry P2 and Tr calibration'}; (dest/'timing.json').write_text(json.dumps(report,indent=2)); print(report,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--outname',default='final'); a.add_argument('--split',default='test',choices=['test','val']); a.add_argument('--scans',type=int,nargs='*',default=[0,200,400,600,800]); a.add_argument('--video',action='store_true'); p=a.parse_args()
    legend_picture(); sensor_check()
    for scan in p.scans: static(scan,p.split,p.outname)
    if p.split=='test': vehicle_details(p.outname)
    if p.video: video(p.outname)
