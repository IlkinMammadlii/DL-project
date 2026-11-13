"""Fetch only requested KITTI image/calibration members over HTTP byte ranges.
No RGB synthesis. ZIP member names bind sequence and scan exactly.
"""
from pathlib import Path
import io,json,zipfile,time
import requests,numpy as np
ROOT=Path(__file__).resolve().parent
BASE='https://s3.eu-central-1.amazonaws.com/avg-kitti/'

class HTTPFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url; self.session=requests.Session(); self.pos=0; self.transferred=0
        r=self.session.get(url,headers={'Range':'bytes=0-0'},timeout=30); r.raise_for_status(); assert r.status_code==206
        self.size=int(r.headers['Content-Range'].split('/')[-1])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset; return self.pos
    def read(self,n=-1):
        if n<0: n=self.size-self.pos
        if n==0: return b''
        end=min(self.pos+n,self.size)-1
        # Query distinguishes range responses in intermediary caches.
        url=self.url+('?' if '?' not in self.url else '&')+f'range={self.pos}-{end}'
        for attempt in range(4):
            try:
                r=self.session.get(url,headers={'Range':f'bytes={self.pos}-{end}'},timeout=60); r.raise_for_status(); assert r.status_code==206,(r.status_code,len(r.content)); data=r.content; assert len(data)==end-self.pos+1; break
            except Exception:
                if attempt==3: raise
                time.sleep(2)
        self.pos+=len(data); self.transferred+=len(data); return data

def get_camera(scans,seq='03'):
    dest=ROOT/'camera'/seq; dest.mkdir(parents=True,exist_ok=True); source=HTTPFile(BASE+'data_odometry_color.zip'); z=zipfile.ZipFile(source)
    for scan in scans:
        target=dest/f'{scan:06d}.png'; member=f'dataset/sequences/{seq}/image_2/{scan:06d}.png'
        if not target.exists(): target.write_bytes(z.read(member)); print('camera',seq,scan,'downloaded',flush=True)
    (ROOT/'camera'/f'download_{seq}.json').write_text(json.dumps({'source':source.url,'archive_bytes':source.size,'bytes_transferred_this_run':source.transferred,'sequence':seq,'scans':list(scans),'note':'image_2 frames read by exact sequence/scan ZIP member'},indent=2))
    for name in ['data_odometry_calib.zip','data_odometry_poses.zip']:
        p=ROOT/'camera'/name
        if not p.exists(): r=requests.get(BASE+name,timeout=40); r.raise_for_status(); p.write_bytes(r.content)
    with zipfile.ZipFile(ROOT/'camera/data_odometry_calib.zip') as z:
        for s in ['03','04','07']:
            for fname in ['calib.txt','times.txt']:
                member=f'dataset/sequences/{s}/{fname}'
                if member in z.namelist(): (ROOT/'camera'/f'{s}_{fname}').write_bytes(z.read(member))

def geo_audit():
    """KITTI poses reset origin per sequence; do not mistake local-frame overlap for GPS overlap."""
    out={'odometry_poses':'Each trajectory is relative to its own first camera frame; these positions cannot establish cross-sequence geographic separation.','raw_gps':{},'conclusion':'Geographic overlap remains possible until GPS tracks can be aligned in a common geodetic frame.'}
    tracks={}
    for seq,date,drive in [('07','2011_09_30','0027'),('06','2011_09_30','0020'),('04','2011_09_30','0016'),('03','2011_09_26','0067')]:
        url=BASE+f'raw_data/{date}_drive_{drive}/{date}_drive_{drive}_sync.zip'
        try:
            f=HTTPFile(url); z=zipfile.ZipFile(f); members=sorted(n for n in z.namelist() if '/oxts/data/' in n and n.endswith('.txt'))
            chosen=members[::10]; gps=np.array([np.fromstring(z.read(n).decode(),sep=' ')[:3] for n in chosen]); assert len(gps)>0
            tracks[seq]=gps; np.save(ROOT/'camera'/f'{seq}_gps.npy',gps); out['raw_gps'][seq]={'source':url,'raw_frames':len(members),'sampled_gps':len(gps),'lat_range':[float(gps[:,0].min()),float(gps[:,0].max())],'lon_range':[float(gps[:,1].min()),float(gps[:,1].max())],'bytes_transferred':f.transferred,'note':'Raw drive associated with odometry sequence; GPS every tenth raw scan.'}
        except Exception as e: out['raw_gps'][seq]={'source':url,'error':repr(e)}
    if len(tracks)>1:
        ref=np.concatenate(list(tracks.values())).mean(0); xy={s:np.column_stack(((g[:,1]-ref[1])*111320*np.cos(np.deg2rad(ref[0])),(g[:,0]-ref[0])*111320)) for s,g in tracks.items()}; distances={}
        for a,b in [('07','04'),('06','04'),('07','03'),('06','03'),('04','03')]:
            if a in xy and b in xy: distances[f'{a}-{b}']=float(np.linalg.norm(xy[a][:,None]-xy[b][None,:],axis=2).min())
        out['minimum_sampled_gps_distance_metres']=distances
        out['conclusion']='Sampled GPS nearest distances are descriptive: not a guarantee against shared views, GPS error, or unsampled route overlap. Missing raw GPS remains an unresolved limitation.'
    (ROOT/'geographic_audit.json').write_text(json.dumps(out,indent=2)); print(out,flush=True)

if __name__=='__main__':
    try: get_camera(sorted(set(list(range(100,300))+[0,200,400,600,800])))
    except Exception as e:
        (ROOT/'camera_failure.json').write_text(json.dumps({'error':repr(e),'note':'Optional imagery failed; core LiDAR pipeline continues.'},indent=2)); print(repr(e),flush=True)
    geo_audit()
