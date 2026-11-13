"""Sequential inference on recorded scans, optionally displaying each result.
This is recorded-file playback, not a connection to a live LiDAR sensor.
"""
import argparse,json,time
import numpy as np,torch
from PIL import Image
from data import ROOT
from infer import load,predict_scene
from visuals import driver,semantic,intensity,panel,BG
def main(checkpoint,start,stop,display):
    m,ck=load(checkpoint); statsfile=checkpoint.parent/'train_stats.json'; stats=json.loads(statsfile.read_text()); timings=[]; root=None
    if display:
        import tkinter as tk
        from PIL import ImageTk
        root=tk.Tk(); root.title('PointNet++: sequential recorded-scan inference'); label=tk.Label(root); label.pack()
    for scan in range(start,stop):
        t=time.perf_counter(); x=np.load(ROOT/'cache/03'/f'{scan:06d}_x.npy'); pred,conf,timing=predict_scene(m,x,stats,scan)
        image=Image.new('RGB',(1240,430),BG); image.paste(panel(driver(x,intensity(x)),'Recorded LiDAR input',f'Sequence 03 / {scan:06d}',(620,430)),(0,0)); image.paste(panel(driver(x,semantic(pred)),'Newly computed PointNet++ output',f"Processing time {timing['wall_seconds']:.2f}s; recorded-file playback",(620,430)),(620,0))
        if root:
            photo=ImageTk.PhotoImage(image); label.configure(image=photo); label.image=photo; root.update()
        timing.update(scan=scan,wall_including_load_and_render=time.perf_counter()-t); timings.append(timing); print(timing,flush=True)
    (ROOT/'stream_timing.json').write_text(json.dumps(timings,indent=2))
    if root: root.destroy()
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=lambda s:ROOT/s,default=ROOT/'checkpoints/best.pt'); p.add_argument('--start',type=int,default=100); p.add_argument('--stop',type=int,default=110); p.add_argument('--display',action='store_true'); a=p.parse_args(); main(a.checkpoint,a.start,a.stop,a.display)
