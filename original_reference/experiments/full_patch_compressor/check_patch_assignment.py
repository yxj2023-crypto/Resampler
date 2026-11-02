"""Audit patch-to-slot argmax occupancy on complete validation cases."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def summarize(assignment, case_id):
    labels = assignment.argmax(-1)
    counts = np.bincount(labels, minlength=assignment.shape[1])
    confidence = assignment[np.arange(len(labels)), labels]
    top_two = np.partition(assignment, -2, axis=-1)[:, -2:]
    margin = top_two[:, 1] - top_two[:, 0]
    slot_rows = []
    for slot, count in enumerate(counts):
        mask = labels == slot
        slot_rows.append(dict(case_id=case_id, slot_id=slot, hard_patch_count=int(count),
                              hard_fraction=float(count / len(labels)),
                              soft_mass=float(assignment[:, slot].mean()),
                              mean_winning_weight=float(confidence[mask].mean()) if count else None,
                              patches_winning_weight_ge_0p5=int((confidence[mask] >= .5).sum()),
                              mean_top1_top2_margin=float(margin[mask].mean()) if count else None))
    case_row = dict(case_id=case_id, patches=len(labels), occupied_slots=int((counts > 0).sum()),
                    empty_slots=np.flatnonzero(counts == 0).tolist(),
                    min_hard_count=int(counts.min()), max_hard_count=int(counts.max()),
                    median_winning_weight=float(np.median(confidence)),
                    fraction_winning_weight_ge_0p5=float((confidence >= .5).mean()),
                    mean_top1_top2_margin=float(margin.mean()), counts=counts.tolist())
    assert counts.sum() == len(labels)
    return case_row, slot_rows


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='results/full_patch_compressor_adam_gc8_20261013')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run = ROOT / args.run_dir
    checkpoint = torch.load(run / 'best.pt', map_location='cpu')
    epoch = checkpoint['epoch']
    output = ROOT / 'reports' / run.name / ('patch_assignment_epoch_%04d' % epoch)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model_name = checkpoint['settings']['model']
    is_perceiver = model_name in ('OrthogonalPerceiver', 'ReconstructionPerceiver')
    model_path = ('experiments/perceiver_orth_rec/model.py' if model_name == 'ReconstructionPerceiver'
                  else 'experiments/perceiver_orthogonal/model.py' if is_perceiver
                  else 'src/modules/resampler.py')
    spec = importlib.util.spec_from_file_location('audited_resampler', run / 'source' / model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if is_perceiver:
        settings = checkpoint['settings']
        model_args = {k: settings[k] for k in (
            'input_dim', 'hidden_dim', 'num_slots', 'self_attn_layers', 'iterations',
            'heads', 'orthogonal_init')}
        if model_name == 'ReconstructionPerceiver':
            model_args['decoder_type'] = settings['decoder_type']
        model = getattr(module, model_name)(**model_args)
    else:
        model = module.SlotAttnResampler()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(args.device)
    manifest = json.loads((run / 'manifest.json').read_text())
    monitor_case = json.loads((run / 'monitor/case.json').read_text())
    epochs = []
    for folder in sorted((run / 'monitor').glob('epoch_*')):
        with np.load(folder / 'matrices.npz') as matrices:
            row, slots = summarize(matrices['assignment'], monitor_case['case_id'])
        row['epoch'] = int(folder.name.split('_')[1])
        epochs.append(row)
        if row['epoch'] == epoch:
            monitor, monitor_slots = row, slots
    write_csv(output / 'monitor_slots.csv', monitor_slots)
    cases, all_slots = [], []
    with torch.no_grad():
        for i, record in enumerate(manifest['val']):
            x = torch.cat([torch.load(path, map_location='cpu') for path in record['paths']]).float().unsqueeze(0)
            assert x.shape[1] == record['patches']
            result = model(x.to(args.device), return_details=True)
            assignment = result['details']['assignment'][0].cpu().numpy()
            row, slots = summarize(assignment, record['case_id'])
            cases.append(row)
            all_slots.extend(slots)
            if (i + 1) % 10 == 0 or i + 1 == len(manifest['val']):
                print('Complete validation cases: %d/%d' % (i + 1, len(manifest['val'])), flush=True)
    write_csv(output / 'validation_slots.csv', all_slots)
    summary = dict(epoch=epoch, monitor=monitor,
                   validation_cases=len(cases), validation_patches=sum(r['patches'] for r in cases),
                   cases_with_all_slots_occupied=sum(not r['empty_slots'] for r in cases),
                   cases_with_empty_slots=sum(bool(r['empty_slots']) for r in cases),
                   occupied_slots_min=min(r['occupied_slots'] for r in cases),
                   occupied_slots_median=float(np.median([r['occupied_slots'] for r in cases])),
                   global_min_slot_count=min(r['min_hard_count'] for r in cases),
                   cases_with_empty_slot_details=[r for r in cases if r['empty_slots']],
                   monitor_epochs_with_empty_slots=[r for r in epochs if r['empty_slots']])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'validation_cases.json').write_text(json.dumps(cases, indent=2))
    (output / 'monitor_epochs.json').write_text(json.dumps(epochs, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), layout='constrained')
    colors = plt.get_cmap('turbo')(np.linspace(0, 1, len(monitor['counts'])))
    axes[0].bar(np.arange(len(monitor['counts'])), monitor['counts'], color=colors)
    axes[0].set(title='%s | epoch %d | patch argmax assignment | %d/%d occupied slots' % (
        monitor['case_id'], epoch, monitor['occupied_slots'], len(monitor['counts'])),
        xlabel='Slot ID', ylabel='Number of assigned patches')
    axes[0].set_xticks(np.arange(len(monitor['counts'])))
    axes[0].grid(axis='y', alpha=.2)
    counts = np.array([r['counts'] for r in cases])
    shown = axes[1].imshow(np.log10(1 + counts), aspect='auto', cmap='viridis', vmin=0)
    for case_index, slot_index in np.argwhere(counts == 0):
        axes[1].plot(slot_index, case_index, 'rx', markersize=5, markeredgewidth=1)
    axes[1].set(title='All %d complete validation cases; red crosses = empty slots' % len(cases),
                xlabel='Slot ID', ylabel='Validation case index (order in validation_cases.json)')
    fig.colorbar(shown, ax=axes[1], label='log10(1 + hard patch count)')
    fig.savefig(output / 'occupancy.png', dpi=150)
    plt.close(fig)
    report = ['# Patch 选择 slot 的占用检查', '',
              '使用 epoch %d 的 best.pt；完整验证病例，无 patch 采样。每个 patch 只分配给软权重最大的 slot。' % epoch, '',
              '监控病例 %s：%d/%d 个 slot 非空，每 slot %d–%d 个 patch。' % (
                  monitor['case_id'], monitor['occupied_slots'], len(monitor['counts']),
                  monitor['min_hard_count'], monitor['max_hard_count']), '',
              '完整验证集：%d 个病例中，%d 个病例全部 slot 非空，%d 个病例存在空 slot。' % (
                  len(cases), summary['cases_with_all_slots_occupied'], summary['cases_with_empty_slots']), '',
              '当前 softmax 和按列归一化没有硬分配的非空约束；非零 soft mass 不代表该 slot 赢得任何 patch。', '',
              '![硬分配占用](occupancy.png)', '',
              '[监控病例逐 slot 数量与置信度](monitor_slots.csv) · '
              '[完整验证集逐 slot 统计](validation_slots.csv) · [汇总](summary.json)']
    (output / 'README.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
