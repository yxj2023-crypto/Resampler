"""Train the current compressor on complete patient bags, without patch sampling."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import FEATURE_DIR, GENE_DIR, SPLIT_DIR
from src.modules.resampler import SlotAttnResampler
from src.utils import set_seed


class FullBags(Dataset):
    def __init__(self, rows, feature_dir):
        self.records = [dict(case_id=row.case_id,
                             paths=[str(feature_dir / (Path(slide).stem + '.pt'))
                                    for slide in row.slide_id])
                        for row in rows.itertuples()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        tensors = [torch.load(path, map_location='cpu') for path in self.records[index]['paths']]
        return torch.cat(tensors, dim=0).float(), index


def geometry(x):
    z = F.normalize(x, dim=-1)
    z = z - z.mean(-2, keepdim=True)
    energy = torch.linalg.svdvals(z).square()
    p = energy / energy.sum(-1, keepdim=True).clamp_min(1e-30)
    return {'effective_rank': float((-(p * p.clamp_min(1e-30).log()).sum(-1)).exp().mean()),
            'pc1': float(p[..., 0].mean())}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {}
    cases = []
    for x, index in loader:
        record = loader.dataset.records[int(index)]
        record['patches'] = int(x.shape[1])
        result = model(x.to(device, non_blocking=True), return_slots=True)
        metrics = {k: float(v) for k, v in result['monitor'].items()}
        metrics.update({'hidden_' + k: v for k, v in geometry(result['slots']).items()})
        metrics.update({'prototype_' + k: v for k, v in geometry(result['prototypes']).items()})
        cases.append(dict(case_id=record['case_id'], patches=record['patches'], **metrics))
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.) + v
    return {k: v / len(loader.dataset) for k, v in totals.items()}, cases


def coordinates(x):
    z = F.normalize(x.double(), dim=-1).numpy()
    z -= z.mean(0)
    _, _, vh = np.linalg.svd(z, full_matrices=False)
    return z @ vh[:2].T


class Monitor:
    def __init__(self, dataset, case_id, output):
        record = next(r for r in dataset.records if r['case_id'] == case_id)
        self.x = torch.cat([torch.load(p, map_location='cpu') for p in record['paths']]).float().unsqueeze(0)
        self.output = output / 'monitor'
        self.output.mkdir()
        self.slide_data = []
        for path in record['paths']:
            path = Path(path)
            root = path.parent.parent
            with h5py.File(root / 'patches' / (path.stem + '.h5'), 'r') as f:
                coords = f['coords'][:]
                patch_size = int(f['coords'].attrs['patch_size'])
                level = int(f['coords'].attrs['patch_level'])
            with openslide.OpenSlide(str(root / 'wsi' / (path.stem + '.svs'))) as slide:
                self.slide_data.append((path.stem, coords, slide.dimensions,
                                        patch_size * slide.level_downsamples[level],
                                        slide.get_thumbnail((1400, 1400)).convert('RGB')))
        assert sum(len(s[1]) for s in self.slide_data) == self.x.shape[1]
        (self.output / 'case.json').write_text(json.dumps(dict(
            case_id=case_id, paths=record['paths'], patches=int(self.x.shape[1]),
            patch_sampling=False, coloring='argmax of full soft assignment; slot IDs are unchanged'), indent=2))

    @torch.no_grad()
    def log(self, model, epoch, device):
        model.eval()
        result = model(self.x.to(device), return_slots=True, return_details=True)
        slots = result['slots'][0].cpu()
        prototypes = result['prototypes'][0].cpu()
        assignment = result['details']['assignment'][0].cpu().numpy()
        mass = result['mass'][0].cpu().numpy()
        folder = self.output / ('epoch_%04d' % epoch)
        folder.mkdir()
        np.savez_compressed(folder / 'matrices.npz', slots=slots.numpy(),
                            prototypes=prototypes.numpy(), assignment=assignment, mass=mass)
        metrics = {k: float(v) for k, v in result['monitor'].items()}
        metrics.update({'hidden_' + k: v for k, v in geometry(slots).items()})
        metrics.update({'prototype_' + k: v for k, v in geometry(prototypes).items()})
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        palette = plt.get_cmap('turbo')(np.linspace(0, 1, len(slots)))[:, :3]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
        for ax, x, title in [(axes[0], slots, 'Hidden slots'),
                             (axes[1], prototypes, 'Real patch barycenters')]:
            xy = coordinates(x)
            ax.scatter(*xy.T, c=palette, s=40 + 160 * mass / mass.max())
            for k, point in enumerate(xy):
                ax.annotate(str(k), point, fontsize=6)
            ax.set_title(title)
            ax.set_aspect('equal', adjustable='datalim')
        im = axes[2].imshow(assignment, aspect='auto', vmin=0, vmax=1)
        axes[2].set(title='All patch → slot weights', xlabel='Slot ID', ylabel='Patch index')
        fig.colorbar(im, ax=axes[2])
        fig.savefig(folder / 'slot_graphs.png', dpi=140)
        plt.close(fig)
        labels = assignment.argmax(-1)
        offset = 0
        for stem, coords, dimensions, extent, thumbnail in self.slide_data:
            scale = np.asarray(thumbnail.size) / np.asarray(dimensions)
            overlay = Image.new('RGBA', thumbnail.size)
            draw = ImageDraw.Draw(overlay)
            for coord, label in zip(coords, labels[offset:offset + len(coords)]):
                origin = coord * scale
                color = tuple((palette[label] * 255).astype(int)) + (140,)
                draw.rectangle((*origin, *(origin + extent * scale)), fill=color)
            fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
            axes[0].imshow(thumbnail)
            axes[0].set_title('Original WSI')
            axes[1].imshow(thumbnail)
            axes[1].imshow(overlay)
            axes[1].set_title('All patches → slot IDs (same colors as graph)')
            for ax in axes:
                ax.axis('off')
            fig.savefig(folder / (stem + '_correspondence.png'), dpi=130)
            plt.close(fig)
            offset += len(coords)
        return metrics


def save_report(output, state, history):
    (output / 'status.json').write_text(json.dumps(state, indent=2))
    (output / 'history.json').write_text(json.dumps(history, indent=2))
    lines = ['# 当前压缩器：全量 patch 独立重建实验', '',
             '本实验重新建立基线；此前实验结论全部撤回，历史记录仅作归档。', '',
             'BLCA fold 0，298 train / 75 val。每次一个病例，拼接该病例全部切片的全部 patch；'
             '训练、验证、监控均不采样，不使用旧缓存。', '',
             '当前 SlotAttnResampler：H=256，50 slots，4 iterations。Adam，lr=3e-4，'
             'weight_decay=1e-5，batch_size=1，gc=8，sub_loss_weight=1.0，无梯度裁剪。'
             '仅重建训练，无基因、GW 或生存损失。每轮随机无放回遍历病例；尾组按实际病例数平均。', '',
             '状态：{}；已完成 {}/{} epochs；参数更新 {} 次。'.format(
                 state['status'], max(0, len(history) - 1), state['epochs'], state['optimizer_steps']), '',
             '误差是各病例等权平均的归一化重建误差，病例均值基线为 1；有效秩在逐 slot L2 '
             '归一化、跨 slot 中心化后计算。epoch 0 为未训练评估。', '',
             '| epoch | 更新次数 | 训练重建误差 | 全量验证 decoder 误差 | 全量验证软重心误差 | hidden 秩 | 分配熵 |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in history:
        v = row['val']
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {:.2f} | {:.4f} |'.format(
            row['epoch'], row['optimizer_steps'],
            '{:.4f}'.format(row['train_loss']) if 'train_loss' in row else '—',
            v['decoder_ratio'], v['barycentric_ratio'], v['hidden_effective_rank'], v['assignment_entropy']))
    lines += ['', '结果：`{}`。`manifest.json` 记录全部病例的来源文件和实际 patch 数；'
              '`monitor/epoch_XXXX/` 保存完整监控病例的软分配、slot 图及 WSI 着色图。'
              '`best.pt` 按全量验证 decoder 误差选择，`last.pt` 保存模型和优化器。'.format(output), '',
              '训练完成前不作最终结论；本轮仅是单折单种子全量重建评估。']
    body = '\n'.join(lines) + '\n'
    (output / 'README.md').write_text(body)
    (ROOT / 'reports' / (output.name + '.md')).write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/full_patch_compressor_adam_gc8_20261013')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    cli = parser.parse_args()
    torch.set_num_threads(4)
    set_seed(cli.seed)
    device = torch.device('cuda')
    output = ROOT / cli.output
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    hashes = {}
    for name in ['src/modules/resampler.py', 'experiments/full_patch_compressor/run.py',
                 'src/utils/utils.py', 'config/config.py']:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    rows = pd.read_pickle(GENE_DIR['blca'])
    split = pd.read_csv(SPLIT_DIR['blca'] / 'splits_0.csv')
    datasets = {s: FullBags(rows[rows.case_id.isin(split[s].dropna())], FEATURE_DIR['blca'])
                for s in ['train', 'val']}
    assert len(datasets['train']) == 298 and len(datasets['val']) == 75
    assert not ({r['case_id'] for r in datasets['train'].records}
                & {r['case_id'] for r in datasets['val'].records})
    loaders = {s: DataLoader(d, batch_size=1, shuffle=(s == 'train'),
                            num_workers=cli.num_workers, pin_memory=True)
               for s, d in datasets.items()}
    settings = dict(model='SlotAttnResampler', input_dim=1024, hidden_dim=256, num_slots=50,
                    slot_iterations=4, batch_size=1, gc=8, optimizer='Adam', lr=3e-4,
                    weight_decay=1e-5, sub_loss_weight=1.0, gradient_clip=None,
                    patch_sampling=False, patch_cache=None, full_slides_per_case=True,
                    case_sampling='shuffle_without_replacement', seed=cli.seed, epochs=cli.epochs,
                    fold=0, train_cases=298, val_cases=75, source_sha256=hashes,
                    remainder_normalization='actual_number_of_cases')
    (output / 'settings.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    model = SlotAttnResampler().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    monitor = Monitor(datasets['val'], 'TCGA-2F-A9KO', output)
    state = dict(status='running', pid=os.getpid(), epochs=cli.epochs, optimizer_steps=0,
                 started=datetime.now().astimezone().isoformat(), current_epoch=0)
    history = []
    best = float('inf')
    save_report(output, state, history)
    writer = SummaryWriter(str(output / 'tensorboard'))
    try:
        for epoch in range(cli.epochs + 1):
            started = time.time()
            state['current_epoch'] = epoch
            train_loss = 0.
            if epoch:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for step, (x, index) in enumerate(loaders['train']):
                    datasets['train'].records[int(index)]['patches'] = int(x.shape[1])
                    group_size = min(8, len(loaders['train']) - (step // 8) * 8)
                    result = model(x.to(device, non_blocking=True))
                    loss = result['loss_rec']
                    train_loss += float(loss.detach())
                    (loss / group_size).backward()
                    if (step + 1) % 8 == 0 or step + 1 == len(loaders['train']):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        state['optimizer_steps'] += 1
                    if step % 25 == 0:
                        state.update(current_batch=step, current_patch_count=int(x.shape[1]))
                        (output / 'status.json').write_text(json.dumps(state, indent=2))
                        print('epoch={} batch={}/{} patches={} loss={:.5f} updates={}'.format(
                            epoch, step + 1, len(loaders['train']), x.shape[1], float(loss.detach()),
                            state['optimizer_steps']), flush=True)
            val, case_metrics = evaluate(model, loaders['val'], device)
            monitored = monitor.log(model, epoch, device)
            row = dict(epoch=epoch, optimizer_steps=state['optimizer_steps'], val=val,
                       monitor=monitored, seconds=time.time() - started)
            if epoch:
                row['train_loss'] = train_loss / len(loaders['train'])
            history.append(row)
            (output / ('val_epoch_%04d.json' % epoch)).write_text(json.dumps(case_metrics, indent=2))
            (output / 'manifest.json').write_text(json.dumps(
                {s: d.records for s, d in datasets.items()}, indent=2))
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                              optimizer_steps=state['optimizer_steps'], settings=settings, val=val)
            torch.save(checkpoint, output / 'last.pt')
            if val['decoder_ratio'] < best:
                best = val['decoder_ratio']
                torch.save(checkpoint, output / 'best.pt')
                state.update(best_epoch=epoch, best_val_decoder_ratio=best)
            for key, value in val.items():
                writer.add_scalar('val/' + key, value, epoch)
            if epoch:
                writer.add_scalar('train/loss_rec', row['train_loss'], epoch)
            writer.flush()
            save_report(output, state, history)
            print('EPOCH {} COMPLETE: {}'.format(epoch, json.dumps(row)), flush=True)
        state.update(status='complete', finished=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save_report(output, state, history)
        writer.close()


if __name__ == '__main__':
    main()
