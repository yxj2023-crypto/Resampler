"""Full-patch, orthogonality-only Perceiver experiment; no decoder or reconstruction loss."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import FEATURE_DIR, GENE_DIR, SPLIT_DIR
from src.utils import set_seed
from src.modules.resampler import SlotAttnResampler
from experiments.full_patch_compressor.run import FullBags, Monitor, evaluate
from experiments.perceiver_orthogonal.model import OrthogonalPerceiver


def save_report(output, state, history, settings):
    (output / 'status.json').write_text(json.dumps(state, indent=2))
    (output / 'history.json').write_text(json.dumps(history, indent=2))
    lines = ['# Perceiver：全图、仅 query 正交损失', '',
             '复用 src/modules/perceiver_layer.py::Perceiver / PerceiverTransLayer，'
             '保持原 CrossAttn 更新方式（不加 latent residual）。', '',
             'BLCA fold 0，298 train / 75 val；所有病例全部切片、全部 patch。'
             'Adam，lr=3e-4，weight_decay=1e-5，batch=1，gc=8，30 epochs，seed=1；'
             '尾组按实际病例数平均，无裁剪。', '',
             '初始 query 正交初始化；Q 为精炼后 query 按列排列并 L2 归一化，'
             'loss=||Q.T @ Q-I_50||_F，病例等权平均，权重 1。无重建损失、无 decoder。'
             '这是 query Gram 正交约束，没有额外加入完整 Barlow Twins 的双视图项。', '',
             'H={}，self-attn 层数={}，共享迭代次数={}，参数 {:,}（slot 基线 {:,}）。'
             'encoder 在原 Linear+LayerNorm 后加 H→H 线性适配层以匹配参数量。'.format(
                 settings['hidden_dim'], settings['self_attn_layers'], settings['iterations'],
                 settings['parameters'], settings['slot_baseline_parameters']), '',
             '状态：{}；完成 {}/{} epochs；{} 次更新。'.format(
                 state['status'], history[-1]['epoch'] if history else 0,
                 state['epochs'], state['optimizer_steps']), '',
             '监控沿用 slot 的 query 图、真实 patch 重心、WSI 空间着色、完整软分配矩阵；'
             '分配由最终 refined query 与 patch 的已有 cross-attn logits 沿 query softmax 得到。'
             '重心误差仅作无梯度诊断。best.pt 按验证正交损失选择；另存 best_barycentric.pt 和 last.pt。', '',
             '| epoch | 训练正交损失 | 验证正交损失 | 软重心误差 | hidden 秩 | 重心秩 | 分配熵 | 非空 query |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in history:
        v = r['val']
        lines.append('| {} | {} | {:.6f} | {:.6f} | {:.2f} | {:.2f} | {:.6f} | {:.2f} |'.format(
            r['epoch'], '%.6f' % r['train_loss'] if 'train_loss' in r else '—',
            v['query_orthogonality'], v['barycentric_ratio'], v['hidden_effective_rank'],
            v['prototype_effective_rank'], v['assignment_entropy'], v['occupied_slots']))
    body = '\n'.join(lines) + '\n'
    (output / 'README.md').write_text(body)
    (ROOT / 'reports' / (output.name + '.md')).write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/perceiver_orth_only_fullpatch_20261015')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--self-attn-layers', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--allow-parameter-mismatch', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(4)
    set_seed(cli.seed)
    device = torch.device('cuda')
    model_args = dict(input_dim=1024, hidden_dim=cli.hidden_dim, num_slots=50,
                      self_attn_layers=cli.self_attn_layers, iterations=cli.iterations,
                      heads=cli.heads, orthogonal_init=not cli.random_init)
    # Count without advancing the training RNG.
    with torch.random.fork_rng(devices=[]):
        nbase = sum(p.numel() for p in SlotAttnResampler().parameters())
        nparams = sum(p.numel() for p in OrthogonalPerceiver(**model_args).parameters())
    if nparams != nbase and not cli.allow_parameter_mismatch:
        parser.error('Parameters {} != baseline {}; adjust dimensions/depth or explicitly '
                     'use --allow-parameter-mismatch'.format(nparams, nbase))
    output = ROOT / cli.output
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ['experiments/perceiver_orthogonal/model.py', 'experiments/perceiver_orthogonal/run.py',
                 'experiments/full_patch_compressor/run.py', 'src/modules/perceiver_layer.py',
                 'src/modules/util_layer.py', 'src/modules/resampler.py', 'src/utils/utils.py', 'config/config.py']:
        target = output / 'source' / name
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
                            num_workers=cli.num_workers, pin_memory=True) for s, d in datasets.items()}
    settings = dict(model='OrthogonalPerceiver', **model_args, parameters=nparams,
                    slot_baseline_parameters=nbase, batch_size=1, gc=8, optimizer='Adam', lr=3e-4,
                    weight_decay=1e-5, loss='frobenius_query_gram', orthogonal_weight=1.0,
                    reconstruction_weight=0.0, decoder=False, gradient_clip=None,
                    query_axis='columns: Q[B,H,K], L2 normalized', orthogonal_target='refined',
                    patch_sampling=False, patch_cache=None, full_slides_per_case=True,
                    case_sampling='shuffle_without_replacement', seed=cli.seed, epochs=cli.epochs,
                    fold=0, train_cases=298, val_cases=75, source_sha256=hashes,
                    remainder_normalization='actual_number_of_cases',
                    checkpoint_selection='minimum_validation_query_orthogonality')
    (output / 'settings.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    model = OrthogonalPerceiver(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    monitor = Monitor(datasets['val'], 'TCGA-2F-A9KO', output)
    state = dict(status='running', pid=os.getpid(), epochs=cli.epochs, optimizer_steps=0,
                 started=datetime.now().astimezone().isoformat(), current_epoch=0)
    history = []
    best, best_bary = float('inf'), float('inf')
    save_report(output, state, history, settings)
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
                    loss = result['loss_orth']
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite orthogonality loss')
                    train_loss += float(loss.detach())
                    (loss / group_size).backward()
                    if (step + 1) % 8 == 0 or step + 1 == len(loaders['train']):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        state['optimizer_steps'] += 1
                    if step % 25 == 0:
                        state.update(current_batch=step, current_patch_count=int(x.shape[1]))
                        (output / 'status.json').write_text(json.dumps(state, indent=2))
                        print('epoch={} batch={}/{} patches={} orth={:.6f} updates={}'.format(
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
            (output / 'manifest.json').write_text(json.dumps({s: d.records for s, d in datasets.items()}, indent=2))
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                              optimizer_steps=state['optimizer_steps'], settings=settings, val=val)
            torch.save(checkpoint, output / 'last.pt')
            if val['query_orthogonality'] < best:
                best = val['query_orthogonality']
                torch.save(checkpoint, output / 'best.pt')
                state.update(best_epoch=epoch, best_val_query_orthogonality=best)
            if val['barycentric_ratio'] < best_bary:
                best_bary = val['barycentric_ratio']
                torch.save(checkpoint, output / 'best_barycentric.pt')
                state.update(best_barycentric_epoch=epoch, best_barycentric_ratio=best_bary)
            for key, value in val.items():
                writer.add_scalar('val/' + key, value, epoch)
            if epoch:
                writer.add_scalar('train/loss_orth', row['train_loss'], epoch)
            writer.flush()
            save_report(output, state, history, settings)
            print('EPOCH {} COMPLETE: {}'.format(epoch, json.dumps(row)), flush=True)
        state.update(status='complete', finished=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save_report(output, state, history, settings)
        writer.close()


if __name__ == '__main__':
    main()
