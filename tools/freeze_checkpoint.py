"""Freeze a newly trained winner using validation only, before held-out inference."""
from pathlib import Path
import argparse,json,shutil,hashlib,time

def freeze(project):
    project=project.resolve()
    marker=project/'selection_freeze.json'
    if marker.exists():
        raise RuntimeError('Selection already frozen. Keep this experiment intact; use a new project folder for another experiment.')
    candidates=[]
    for name in ['pointnet2_ssg_07','pointnet2_ssg_06_07']:
        run=project/'runs'/name
        if (run/'complete.json').exists():
            for required in ['best.pt','best_validation.json','train_stats.json']:
                if not (run/required).is_file():raise FileNotFoundError(run/required)
            candidates.append(run)
    if not candidates:raise RuntimeError('No completed training run found.')
    scores={r.name:json.loads((r/'best_validation.json').read_text())['miou19'] for r in candidates}
    winner=max(candidates,key=lambda r:scores[r.name])
    dest=project/'checkpoints';dest.mkdir(exist_ok=True)
    if (dest/'best.pt').exists():raise RuntimeError('Checkpoint destination already exists; refusing to overwrite.')
    shutil.copy2(winner/'best.pt',dest/'best.pt')
    shutil.copy2(winner/'train_stats.json',dest/'train_stats.json')
    metadata={'selected_run':winner.name,'criterion':'Fixed 256-block validation mIoU19 only; no held-out inference before freeze','candidates':scores,'checkpoint_sha256':hashlib.sha256((dest/'best.pt').read_bytes()).hexdigest(),'frozen_unix':time.time()}
    marker.write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('source/pointnet2'));a=p.parse_args();freeze(a.project)
