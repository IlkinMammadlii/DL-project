"""Self-contained reproducible data download with pinned archive SHA-256 checks."""
from pathlib import Path
import argparse,subprocess,hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent
HASHES={'03':'0a1042862ffda32e7951ec42bdb22bf789656d1d4586066ca61a7f42bc83c78f','04':'b7052964fa5edfab70533ee5a0c961c2607a3bd09b83fe529dd9dfbfe30741f2','07':'46c5c49c457062126eb9a8f0fa495a1cd2cbbd0670e283180f279a3ba5374784','06':'7d4d61ceed095f9b808972ed911f751aecf24199c20b0a28cdff8c7f4113ba95'}
def main(extra=False):
    folder=ROOT/'archives'; folder.mkdir(exist_ok=True); results={}
    for seq in (['06'] if extra else ['07','04','03']):
        p=folder/f'{seq}.tar.gz'
        if not p.exists(): subprocess.run(['curl.exe','-L','--fail','--retry','5','-o',str(p),f'https://huggingface.co/datasets/larryshaw0079/SemanticKITTI-SP/resolve/main/{seq}.tar.gz?download=true'],check=True)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
        assert h.hexdigest()==HASHES[seq],p
        results[seq]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}; print(seq,'verified',flush=True)
    (ROOT/'archive_hashes.json').write_text(json.dumps(results,indent=2))
    if not (ROOT/'semantic-kitti.yaml').exists(): urllib.request.urlretrieve('https://raw.githubusercontent.com/PRBonn/semantic-kitti-api/master/config/semantic-kitti.yaml',ROOT/'semantic-kitti.yaml')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--extra06',action='store_true'); main(p.parse_args().extra06)
