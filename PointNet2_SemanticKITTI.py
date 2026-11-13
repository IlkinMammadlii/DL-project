"""Understanding Driving Scenes with PointNet++

Single-file consolidation of the actual project architecture, dataset class and
training loop. This packaging was made after training; the recorded experiments
used the original modular files. No pretrained weights or dataset are embedded.

Quick inspection (no training):
    python PointNet2_SemanticKITTI.py --inspect-model
Training using the original prepared project data:
    python PointNet2_SemanticKITTI.py --data-root PATH_TO_POINTNET2 --phase main
    python PointNet2_SemanticKITTI.py --data-root PATH_TO_POINTNET2 --phase expanded --warmstart PATH_TO_MAIN_BEST_PT

Inputs [batch,4,4096]: XYZ and remission; outputs [batch,19,4096]: raw logits.
Runtime: Python 3.11; torch 2.11.0+cu128, numpy 2.4.6, psutil 7.2.2.
Training requires CUDA and the prepared cache, block indices and statistics.
Dataset download/preparation and full-scene inference remain in the source ZIP.
Code ancestry: yanx27/Pointnet_Pointnet2_pytorch, commit
eb64fe0b4c24055559cea26299cb485dcb43d8dd (MIT). Original notice below.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, json, time, random, sys, platform, subprocess
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import psutil

ROOT = Path(__file__).resolve().parent

# MIT License
# 
# Copyright (c) 2019 benny
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# 1. POINTNET++ LOCAL GROUPING AND FEATURE PROPAGATION
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return (new_xyz, new_points, grouped_xyz, fps_idx)
    else:
        return (new_xyz, new_points)

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return (new_xyz, new_points)

class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return (new_xyz, new_points)

class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = (dists[:, :, :3], idx[:, :, :3])
            dist_recip = 1.0 / (dists + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

# 2. OUR GEOMETRY ADAPTERS AND 19-CLASS MODEL
@torch.jit.script
def fps(xyz: torch.Tensor, npoint: int):
    b, n, _ = xyz.shape
    out = torch.empty((b, npoint), dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 10000000000.0, device=xyz.device)
    farthest = ((xyz - xyz.mean(1, keepdim=True)) ** 2).sum(-1).argmax(-1)
    batch = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        out[:, i] = farthest
        centre = xyz[batch, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - centre) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return out

def distance(src, dst):
    with torch.autocast(device_type=src.device.type, enabled=False):
        return torch.cdist(src.float(), dst.float(), p=2, compute_mode='donot_use_mm_for_euclid_dist').square()

def ball_query(radius, nsample, xyz, new_xyz):
    d = distance(new_xyz, xyz)
    vals, idx = d.topk(min(nsample, xyz.shape[1]), dim=-1, largest=False, sorted=True)
    idx = torch.where(vals <= radius * radius, idx, idx[:, :, :1])
    return idx

class PointNet2(nn.Module):

    def __init__(self, centroids=(512, 128, 32, 8), radii=(0.8, 1.6, 3.2, 6.4), remission=True):
        super().__init__()
        self.remission = remission
        self.sa1 = PointNetSetAbstraction(centroids[0], radii[0], 32, 7 if remission else 6, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(centroids[1], radii[1], 32, 67, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(centroids[2], radii[2], 32, 131, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(centroids[3], radii[3], 32, 259, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(132 if remission else 131, [128, 128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Conv1d(128, 19, 1))

    def forward(self, x):
        xyz = x[:, :3].float()
        feat = x.clone()
        feat[:, :3] = feat[:, :3] / torch.tensor([8.0, 8.0, 5.0], device=x.device).view(1, 3, 1)
        a, fa = self.sa1(xyz, feat)
        b, fb = self.sa2(a, fa)
        c, fc = self.sa3(b, fb)
        d, fd = self.sa4(c, fc)
        fc = self.fp4(c, d, fc, fd)
        fb = self.fp3(b, c, fb, fc)
        fa = self.fp2(a, b, fa, fb)
        out = self.fp1(xyz, a, feat, fa)
        return self.head(out)

# Use the same geometry overrides as the original modular model.
farthest_point_sample = fps
square_distance = distance
query_ball_point = ball_query

# 3. DATASET: BLOCK SAMPLING, NORMALIZATION AND AUGMENTATION
class Blocks(Dataset):

    def __init__(self, split='train', points=4096, seed=42, limit=None, augment=True, stats_name='train'):
        self.blocks = json.loads((ROOT / f'{split}_blocks.json').read_text())
        if limit and len(self.blocks) > limit:
            self.blocks = [self.blocks[i] for i in np.linspace(0, len(self.blocks) - 1, limit, dtype=int)]
        self.points = points
        self.seed = seed
        self.epoch = 0
        self.augment = augment and split.startswith('train')
        self.cache = OrderedDict()
        self.stats = json.loads((ROOT / f'{stats_name}_stats.json').read_text())

    def __len__(self):
        return len(self.blocks)

    def arrays(self, b):
        key = (b['sequence'], b['scan'], b['shift'])
        if key not in self.cache:
            prefix = ROOT / 'cache' / b['sequence'] / f"{b['scan']:06d}"
            self.cache[key] = (np.load(str(prefix) + '_x.npy', mmap_mode='r'), np.load(str(prefix) + '_y.npy', mmap_mode='r'), np.load(str(prefix) + f"_order{b['shift']}.npy", mmap_mode='r'))
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, i):
        b = self.blocks[i]
        x, y, order = self.arrays(b)
        rng = np.random.default_rng(self.seed + i * 1009 + self.epoch * 1000003)
        ids = order[b['start']:b['stop']]
        if self.augment and rng.random() < 0.5:
            ids = rng.choice(ids, max(64, int(len(ids) * rng.uniform(0.7, 1))), replace=False)
        ids = rng.choice(ids, self.points, replace=len(ids) < self.points)
        feat = np.array(x[ids], copy=True)
        target = np.array(y[ids], np.int64)
        feat[:, 0] -= b['cx']
        feat[:, 1] -= b['cy']
        feat[:, 3] = (feat[:, 3] - self.stats['remission_mean']) / self.stats['remission_std']
        if self.augment:
            theta = rng.uniform(-np.pi, np.pi)
            c, s = (np.cos(theta), np.sin(theta))
            feat[:, :2] = feat[:, :2] @ np.array([[c, -s], [s, c]], np.float32)
            if rng.random() < 0.5:
                feat[:, 0] *= -1
            feat[:, :3] *= rng.uniform(0.95, 1.05)
            feat[:, :3] += np.clip(rng.normal(0, 0.01, (len(feat), 3)), -0.03, 0.03).astype(np.float32)
            feat[:, 3] += rng.normal(0, 0.02, len(feat)).astype(np.float32)
        return (torch.from_numpy(feat.T.copy()), torch.from_numpy(target))

def class_weights(dataset='train'):
    counts = np.array(json.loads((ROOT / f'{dataset}_stats.json').read_text())['counts'])
    f = counts / max(counts.sum(), 1)
    w = np.clip(1 / np.sqrt(np.maximum(f, 0.0001)), 1, 10)
    w[counts == 0] = 0
    w /= w[w > 0].mean()
    return w.astype(np.float32)

# 4. REAL TRAINING LOOP, VALIDATION METRICS AND CHECKPOINTS
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def scores(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    gt = cm.sum(1)
    pred = cm.sum(0)
    union = gt + pred - tp

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    iou = div(tp, union)
    return {'miou19': float(iou.mean()), 'miou_present_union': float(iou[union > 0].mean()) if (union > 0).any() else 0.0, 'accuracy': float(tp.sum() / max(gt.sum(), 1)), 'iou': iou.tolist(), 'precision': div(tp, pred).tolist(), 'recall': div(tp, gt).tolist(), 'f1': div(2 * tp, gt + pred).tolist(), 'support': gt.tolist(), 'predicted_support': pred.tolist(), 'confusion': cm.tolist(), 'zero_union': np.flatnonzero(union == 0).tolist()}

@torch.no_grad()
def evaluate(m, loader):
    m.eval()
    cm = np.zeros((19, 19), np.int64)
    losses = []
    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        # Mixed-precision forward computation.
        with torch.autocast('cuda', dtype=torch.float16):
            logits = m(x)
        p = logits.argmax(1)
        v = y >= 0
        cm += torch.bincount(19 * y[v] + p[v], minlength=361).view(19, 19).cpu().numpy()
    return scores(cm)

def state_dict(m, opt, scheduler, scaler, cfg, epoch, step, best, history, seconds):
    return {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'config': cfg, 'epoch': epoch, 'step': step, 'best_val_miou19': best, 'history': history, 'training_wall_seconds': seconds, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}

def train(cfg, resume=None, warmstart=None):
    seed_all(cfg['seed'])
    run = ROOT / 'submission_runs' / cfg['name']
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.json').write_text(json.dumps(cfg, indent=2))
    m = PointNet2().cuda()
    # AdamW uses the learning rate and weight decay shown in section 5.
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=cfg['min_lr'])
    scaler = torch.amp.GradScaler('cuda')
    dataset = cfg.get('dataset', 'train')
    trainset = Blocks(dataset, cfg['points'], cfg['seed'], cfg.get('train_limit'), True, stats_name=dataset)
    valset = Blocks('val', cfg['points'], 123, cfg['val_blocks'], False, stats_name=dataset)
    (run / 'train_stats.json').write_text(json.dumps(trainset.stats, indent=2))
    weights = torch.from_numpy(class_weights(dataset)).cuda()
    # LOSS: class-weighted per-point cross entropy. Ignore target -1.
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-1, label_smoothing=cfg.get('label_smoothing', 0.0))
    history = []
    best = -1.0
    start_epoch = 0
    totalsteps = 0
    previous_seconds = 0.0
    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=True)
        m.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        history = ck['history']
        best = ck['best_val_miou19']
        start_epoch = ck['epoch']
        totalsteps = ck['step']
        previous_seconds = ck['training_wall_seconds']
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif warmstart:
        ck = torch.load(warmstart, map_location='cpu', weights_only=True)
        m.load_state_dict(ck['model'])
        (run / 'warmstart.json').write_text(json.dumps({'checkpoint': str(Path(warmstart).resolve()), 'epoch': ck['epoch'], 'validation_miou19': ck['best_val_miou19']}, indent=2))
    val_loader = DataLoader(valset, batch_size=cfg['batch'], num_workers=0)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    (run / 'class_weights.json').write_text(json.dumps(weights.cpu().tolist()))
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': str(torch.cuda.get_device_properties(0)), 'ram': dict(psutil.virtual_memory()._asdict()), 'command': subprocess.list2cmdline(sys.argv), 'parameters': sum((p.numel() for p in m.parameters())), 'train_blocks': len(trainset), 'validation_blocks': len(valset)}
    (run / 'environment.json').write_text(json.dumps(env, indent=2))
    try:
        for epoch in range(start_epoch, cfg['epochs']):
            trainset.epoch = epoch
            gen = torch.Generator().manual_seed(cfg['seed'] + epoch)
            # BATCH SIZE: cfg[batch]=4 blocks, with 4096 sampled points per block.
            loader = DataLoader(trainset, batch_size=cfg['batch'], shuffle=True, generator=gen, num_workers=0, drop_last=False)
            m.train()
            losses = []
            epochstart = time.perf_counter()
            lastlog = time.perf_counter()
            nsteps = 0
            for step, (x, y) in enumerate(loader):
                if cfg.get('max_steps_per_epoch') and step >= cfg['max_steps_per_epoch']:
                    break
                if not (y >= 0).any():
                    continue
                x = x.cuda()
                y = y.cuda()
                # Clear old gradients before this batch.
                opt.zero_grad(set_to_none=True)
                # Mixed-precision forward computation.
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = criterion(m(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                # BACKPROPAGATION: compute model-parameter gradients.
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
                # OPTIMIZER STEP: apply AdamW update unless AMP detects invalid gradients.
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss))
                totalsteps += 1
                nsteps += 1
                if time.perf_counter() - lastlog > 30:
                    progress = {'epoch': epoch + 1, 'step_in_epoch': step + 1, 'total_steps': totalsteps, 'loss_recent': float(np.mean(losses[-50:])), 'elapsed_seconds': previous_seconds + time.perf_counter() - start, 'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2 ** 20}
                    (run / 'progress.json').write_text(json.dumps(progress, indent=2))
                    print(progress, flush=True)
                    lastlog = time.perf_counter()
            torch.cuda.synchronize()
            trainsec = time.perf_counter() - epochstart
            # VALIDATION: evaluate without updating model weights.
            r = evaluate(m, val_loader)
            # Change the learning rate for the next epoch.
            scheduler.step()
            row = {'epoch': epoch + 1, 'steps': nsteps, 'total_steps': totalsteps, 'loss': float(np.mean(losses)), 'val_miou19': r['miou19'], 'val_accuracy': r['accuracy'], 'val_car_iou': r['iou'][0], 'val_road_iou': r['iou'][8], 'train_seconds': trainsec, 'lr_next': scheduler.get_last_lr()[0], 'wall_seconds': previous_seconds + time.perf_counter() - start, 'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2 ** 20}
            history.append(row)
            (run / 'history.json').write_text(json.dumps(history, indent=2))
            print(row, flush=True)
            # Select checkpoints using validation mIoU; never use held-out test scores.
            if r['miou19'] > best:
                best = r['miou19']
                ck = state_dict(m, opt, scheduler, scaler, cfg, epoch + 1, totalsteps, best, history, previous_seconds + time.perf_counter() - start)
                torch.save(ck, run / 'best.pt')
                (run / 'best_validation.json').write_text(json.dumps(r, indent=2))
            ck = state_dict(m, opt, scheduler, scaler, cfg, epoch + 1, totalsteps, best, history, previous_seconds + time.perf_counter() - start)
            torch.save(ck, run / 'last.tmp')
            os.replace(run / 'last.tmp', run / 'last.pt')
            best_epoch = max(history, key=lambda h: h['val_miou19'])['epoch']
            if epoch + 1 >= cfg.get('minimum_epochs', cfg['epochs']) and epoch + 1 - best_epoch >= cfg.get('early_stop_patience', cfg['epochs']):
                print('Validation-only early stopping', flush=True)
                break
        (run / 'complete.json').write_text(json.dumps({'epochs': history[-1]['epoch'], 'planned_maximum_epochs': cfg['epochs'], 'updates': totalsteps, 'wall_seconds': previous_seconds + time.perf_counter() - start, 'best_validation_miou19': best, 'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2 ** 20}, indent=2))
    except BaseException as e:
        (run / 'failure.json').write_text(json.dumps({'error': repr(e), 'time': time.time(), 'last_completed_epoch': history[-1]['epoch'] if history else 0}, indent=2))
        raise

def overfit():
    seed_all(19)
    ds = Blocks('train', 4096, 19, augment=False)
    chosen = []
    for i, b in enumerate(ds.blocks):
        x, y = ds[i]
        if len(torch.unique(y[y >= 0])) >= 5 and (y == 0).sum() > 50:
            chosen.append((x, y, b))
        if len(chosen) == 2:
            break
    assert len(chosen) == 2
    x = torch.stack([a[0] for a in chosen]).cuda()
    y = torch.stack([a[1] for a in chosen]).cuda()
    m = PointNet2().cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=0.003)
    scaler = torch.amp.GradScaler('cuda')
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    losses = []
    t = time.perf_counter()
    for step in range(240):
        # Clear old gradients before this batch.
        opt.zero_grad(set_to_none=True)
        # Mixed-precision forward computation.
        with torch.autocast('cuda', dtype=torch.float16):
            loss = lossfn(m(x), y)
        # BACKPROPAGATION: compute model-parameter gradients.
        scaler.scale(loss).backward()
        # OPTIMIZER STEP: apply AdamW update unless AMP detects invalid gradients.
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))
        if step % 40 == 0:
            print('overfit', step, losses[-1], flush=True)
    m.eval()
    with torch.no_grad():
        p = m(x).argmax(1)
        v = y >= 0
        acc = float((p[v] == y[v]).float().mean())
    out = {'initial_loss': losses[0], 'final_loss': losses[-1], 'accuracy': acc, 'seconds': time.perf_counter() - t, 'steps': len(losses), 'blocks': [a[2] for a in chosen], 'classes': torch.unique(y[y >= 0]).cpu().tolist(), 'losses': losses}
    (ROOT / 'submission_runs' / 'overfit_diagnostic' / 'overfit.json').write_text(json.dumps(out, indent=2))
    torch.save({'model': m.state_dict(), 'purpose': 'overfit diagnostic only'}, ROOT / 'submission_runs' / 'overfit_diagnostic' / 'overfit.pt')
    print({k: v for k, v in out.items() if k != 'losses'}, flush=True)

# 5. EXACT HYPERPARAMETERS FROM THE COMPLETED EXPERIMENTS
MAIN_CONFIG = {'name': 'pointnet2_ssg_07',
 'seed': 42,
 'points': 4096,
 'batch': 4,
 'epochs': 40,
 'lr': 0.001,
 'min_lr': 1e-05,
 'weight_decay': 0.0001,
 'label_smoothing': 0.0,
 'val_blocks': 256,
 'max_steps_per_epoch': None,
 'train_limit': None,
 'early_stop_patience': 8,
 'minimum_epochs': 20,
 'comment': 'Full passes through all 42,113 overlapping 16m blocks of all 1,101 '
            'sequence-07 scans. Up to 421,160 optimiser updates; validation-only early '
            'stopping after at least 20 full epochs.'}
EXPANDED_CONFIG = {'name': 'pointnet2_ssg_06_07',
 'dataset': 'train_joint',
 'seed': 42,
 'points': 4096,
 'batch': 4,
 'epochs': 10,
 'lr': 0.0005,
 'min_lr': 1e-05,
 'weight_decay': 0.0001,
 'label_smoothing': 0.0,
 'val_blocks': 256,
 'max_steps_per_epoch': None,
 'train_limit': None,
 'early_stop_patience': 10,
 'minimum_epochs': 10,
 'comment': 'Conditional expansion: warm-start from sequence-07 best; 10 full passes '
            'through sequences 06+07, only if remaining measured budget allows. Fresh '
            'optimiser/schedule; training-only joint statistics.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, help='Prepared project folder containing cache/, *_blocks.json and *_stats.json')
    parser.add_argument('--phase', choices=['main', 'expanded'], default='main')
    parser.add_argument('--config', type=Path, help='Optional explicit JSON configuration instead of embedded phase config')
    parser.add_argument('--resume', help='Restore a completed epoch plus optimizer, scheduler and scaler state')
    parser.add_argument('--warmstart', help='Load model weights only; start a fresh optimizer and schedule')
    parser.add_argument('--overfit', action='store_true', help='Run the separate two-block diagnostic, not serious training')
    parser.add_argument('--inspect-model', action='store_true', help='Print model and configuration; does not train or need CUDA')
    args = parser.parse_args()
    if args.overfit and (args.resume or args.warmstart or args.config or args.phase != 'main' or args.inspect_model):
        parser.error('--overfit is a separate fresh diagnostic; do not combine it with --resume, --warmstart, --config, --phase expanded or --inspect-model.')
    cfg = json.loads(args.config.read_text()) if args.config else dict(MAIN_CONFIG if args.phase == 'main' else EXPANDED_CONFIG)
    if args.inspect_model:
        model = PointNet2()
        print(model)
        print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Configuration:', json.dumps(cfg, indent=2))
        return
    if args.resume and args.warmstart:
        parser.error('Use --resume or --warmstart, not both.')
    if args.phase == 'expanded' and not (args.warmstart or args.resume):
        parser.error('Expanded phase needs the main best weights via --warmstart, or its own --resume checkpoint.')
    if args.data_root is None:
        parser.error('--data-root is required for training; use --inspect-model to inspect without training.')
    global ROOT
    ROOT = args.data_root.resolve()
    dataset = cfg.get('dataset', 'train')
    required = [ROOT/'cache', ROOT/f'{dataset}_blocks.json', ROOT/f'{dataset}_stats.json', ROOT/'val_blocks.json']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        parser.error('Missing prepared data: ' + ', '.join(missing))
    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU. Code inspection works with --inspect-model.')
    if args.overfit:
        (ROOT / 'submission_runs' / 'overfit_diagnostic').mkdir(parents=True, exist_ok=True)
        overfit()
    else:
        train(cfg, args.resume, args.warmstart)

if __name__ == '__main__':
    main()
